function dist_wass(p, q)
    d = length(p);  # Suponemos que p y q tienen la misma longitud, que define d

    # Crear la matriz de costos `c`, tal que c[j, k] = 1 si j ≠ k y c[j, k] = 0 si j = k
    c = zeros(Float64, d, d)  # Inicializar la matriz de costos
    for j in 1:d, k in 1:d
        c[j, k] = (j != k) ? 1.0 : 0.0;  # Asignar costos según la condición
    end

    # Escalar p y q por 1e6
    p = 1e6 * p;
    q = 1e6 * q;

    # Definir el modelo del solver
    model = Model(Clp.Optimizer);
    set_silent(model);  # Silencia el solver para no mostrar salida en consola

    # Definir las variables y restricciones
    @variable(model, w[1:d, 1:d] >= 0);  # Tensor w de tamaño d x d

    # Restricciones: sum(w[j, k], k) = p[j]
    for j in 1:d
        @constraint(model, sum(w[j, k] for k in 1:d) == p[j]);
    end

    # Restricciones: sum(w[j, k], j) = q[k]
    for k in 1:d
        @constraint(model, sum(w[j, k] for j in 1:d) == q[k]);
    end

    # Definir la función objetivo
    @objective(model, Min, sum(c[j, k] * w[j, k] for j in 1:d, k in 1:d));

    # Resolver el modelo
    optimize!(model);

    # Obtener la distancia Wasserstein
    dist = sum(c[j, k] * value(w[j, k]) for j in 1:d, k in 1:d);

    # Ajustar por el factor 1e6
    dist /= 1e6;
    # Obtener los valores de w
    # w_val = [value(w[j, k]) for j in 1:d, k in 1:d];
    # w_val = reshape(w_val, (d, d));
    # , w_val/1e6
    # Se divide entre 2
    return dist/2
end

# Definir la función moda_wass
function moda_wass(P, lambdas=nothing)
    n, d = size(P);  # Tamaño de P
    P = 1e6 * P;  # Escalar P

    # Asignar valores por defecto a lambdas si es necesario
    if lambdas === nothing
        lambdas = fill(1/n, n);  # Crear un vector de longitud n con valores 1/n
    end

    # Crear la matriz de costos `c`
    c = zeros(Float64, d, d)
    for j in 1:d, k in 1:d
        c[j, k] = (j != k) ? 1.0 : 0.0;
    end

    # Definir el modelo de optimización
    model = Model(Clp.Optimizer);
    set_silent(model);

    # Definir las variables y restricciones
    @variable(model, w[1:n, 1:d, 1:d] >= 0);

    # Restricciones: sum(w[i, j, k], k) = P[j, i]
    for i in 1:n, j in 1:d
        @constraint(model, sum(w[i, j, k] for k in 1:d) == P[i, j]);
    end

    # Restricciones: sum(w[i, j, k], j) = q[k]
    for i in 2:n, k in 1:d
        @constraint(model, sum(w[i, j, k] for j in 1:d) == sum(w[1, j, k] for j in 1:d));
    end

    # Definir la función objetivo
    @objective(model, Min, sum(lambdas[i] * c[j, k] * w[i, j, k] for i in 1:n, j in 1:d, k in 1:d));

    # Resolver el modelo
    @time optimize!(model);

    # Obtener los valores de w
    w_val = [value(w[i, j, k]) for i in 1:n, j in 1:d, k in 1:d];
    w_val = reshape(w_val, (n, d, d));

    # Obtener q_val
    q_val = [sum(value(w[1, j, k]) for j in 1:d) for k in 1:d];
    q_val /= 1e6;
    q_val /= sum(q_val);

    # Obtener el valor de la función objetivo (costo total)
    cost_value = objective_value(model);  # Obtener el valor de la función objetivo
    
   return q_val, w_val/1e6, cost_value/1e6
end

# Distancia de Hamming multivariada
function dist_wass_mult(concepto1, concepto2, variables)
    suma = 0
    for var in variables
        suma += dist_wass(concepto1[var], concepto2[var])  
    end
    return suma
end


# Moda multivariada
function moda_wass_mult(diccionario, lambdas = nothing)
    # Crear diccionario vacío para almacenar las modas de cada variable
    modas = Dict();
    variables =  collect(keys(diccionario))
    for var in variables
        # println(var)
        P = diccionario[var];
        moda, conjunta, costo = moda_wass(P, lambdas);
        modas[var] = moda;
    end
    return modas
end

function media_mult(diccionario)
    # Crear diccionario vacío para almacenar los vectores de medias de cada variable
    medias = Dict()
    variables = collect(keys(diccionario))
    
    for var in variables
        # println(var)
        P = diccionario[var]
        medias[var] = mean(P, dims=1)  # Calcular la media por columnas
    end
    
    return medias
end

function selecciona_indices(diccionario, indices)
    # Crear diccionario vacío 
    seleccion = Dict();
    variables = collect(keys(diccionario));
    
    for var in variables
        # println(var);
        P = diccionario[var];
        seleccion[var] = P[indices,:];
    end
    
    return seleccion
end

function inercia_intra_wass(modas, diccionario, clusters, variables, K, n, lambdas) 
    inercia_W = 0
    for k in 1:K
        moda_k = modas[k]
        indices_cluster_k = findall(c -> c == k, clusters)
        for i in indices_cluster_k
            concepto_i = selecciona_indices(diccionario, i)
            dist_ik = dist_wass_mult(concepto_i, moda_k, variables)
            inercia_W += dist_ik * lambdas[i]
        end
    end
    return inercia_W
end

function k_modas_wass(diccionario, K, tol, n_iter_max, lambdas = nothing)
    # Obtener las variables del diccionario
    variables = collect(keys(diccionario))
    # Determinar el tamaño n, suponiendo que todos los arrays en el diccionario tienen la misma longitud
    n = size(diccionario[variables[1]])[1]
    # Si lambdas es nothing, inicializarlo como un vector de tamaño n con valores iguales a 1/n
    lambdas = isnothing(lambdas) ? fill(1/n, n) : lambdas
    # Crear el diccionario de modas
    modas = Dict()
    
    # Escogencia aleatoria inicial de índices
    indices_aleatorios = randperm(n)[1:K];
    
    for k in 1:K
        indice = indices_aleatorios[k];
        sub_dict = Dict();
        
        for var in variables
            sub_dict[var] = diccionario[var][indice, :];
        end
        
        modas[k] = sub_dict;
    end

    # Inicialización de los grupos
    clusters = [0 for i in 1:n];

    error_anterior = 0
    cambio_error = tol + 1;
    n_iter = 0; 
    error_actual = Inf;  # Inicializa error_actual
    continuar = true;  # Variable de control del ciclo

    # Vector para almacenar los cambios de error en cada iteración
    cambio_error_vec = Float64[]; 

    while continuar
        # Fase de asignación: se asigna cada concepto al grupo más cercano
        for i in 1:n
            concepto_i = selecciona_indices(diccionario, i);  
            cluster_actual_i = 0;
            dist_ant = Inf;
            for k in 1:K
                dist_k = dist_wass_mult(concepto_i, modas[k], variables);            
                # Comparar y asignar el clúster más cercano
                if dist_k < dist_ant
                    dist_ant = dist_k;  # Actualiza la distancia más pequeña
                    cluster_actual_i = k;  # Asigna el índice del clúster
                end
            end
            # Asignar el clúster encontrado
            clusters[i] = cluster_actual_i;
        end
        
        # Calcular la inercia intra-clase
        error_actual = inercia_intra_wass(modas, diccionario, clusters, variables, K, n, lambdas);
        
        # Fase de representación: se recalculan las modas
        for k in 1:K
            # Obtener los índices de los elementos en el clúster k
            indices_cluster_k = findall(c -> c == k, clusters)
            
            # Seleccionar los conceptos correspondientes a esos índices
            conceptos_cluster_k = selecciona_indices(diccionario, indices_cluster_k)
            
            # Extraer los lambdas para estos índices
            lambdas_cluster_k = lambdas[indices_cluster_k]
            
            # Normalizar lambdas para que sumen 1
            lambdas_cluster_k /= sum(lambdas_cluster_k)
            
            # Calcular la moda ponderada utilizando los lambdas
            moda_cluster_k = moda_wass_mult(conceptos_cluster_k, lambdas_cluster_k)
            
            # Guardar la moda recalculada para el clúster k
            modas[k] = moda_cluster_k
        end


        
         # Si n_iter es 0, entonces el cambio_error es igual a error_actual
         if n_iter == 0
            cambio_error = error_actual;
        else
            cambio_error = error_anterior - error_actual;
        end
        push!(cambio_error_vec, cambio_error);  # Descomentar para almacenar el cambio de error en cada iteración

        # Actualizar si el ciclo debe continuar
        if n_iter >= n_iter_max || 0 <= cambio_error <= tol
            continuar = false;  # Detener el ciclo si se alcanza el número máximo de iteraciones o el cambio de error es pequeño
        else
            # Actualiza error_anterior
            error_anterior = error_actual; 
            # Incrementa el contador de iteraciones
            n_iter += 1;
        end
        
        # Impresión formateada
        # println("Iteración: ", n_iter, ", Inercia intra-clase: ", error_actual);
    end
    
    return clusters, modas, error_actual #, cambio_error_vec;  # Retornar también el vector de cambios de error si es necesario
end
