"""
Optimización usando gradiente descendente - Regresión polinomial
-----------------------------------------------------------------------------------------

En este laboratio se estimarán los parámetros óptimos de un modelo de regresión 
polinomial de grado `n`.

"""

#print("Estoy aca1")
def pregunta_01():
    """
    Complete el código presentado a continuación.
    """
    # Importe pandas
    import pandas as pd

    # Importe PolynomialFeatures
    from sklearn.preprocessing import PolynomialFeatures

    # Cargue el dataset `data.csv`
    data = pd.read_csv('data.csv')

    # Cree un objeto de tipo `PolynomialFeatures` con grado `2`
    poly =  PolynomialFeatures(2) #.fit_transform(data) #___.___(___)

    # Transforme la columna `x` del dataset `data` usando el objeto `poly`
    #print(data[["x"]])
    x_poly = poly.fit_transform(data[["x"]])
    #print(x_poly)

    return x_poly, data.y


#print("Estoy aca2")
def pregunta_02():

    # Importe numpy
    import numpy as np

    x_poly, y = pregunta_01()

    #print ("y",y)

    # Fije la tasa de aprendizaje en 0.0001 y el número de iteraciones en 1000
    learning_rate = 0.0001
    n_iterations = 500

    # Defina el parámetro inicial `params` como un arreglo de tamaño 3 con ceros
    #params = np.zeros(np.shape[2])
    params = np.zeros(3)
    params = np.array([0.0,0.0,0.0])

    #print(params)
    x = [ elem[1] for elem in x_poly]
    #print( x)
    y_pred = np.polyval(params, x) 
    #print("y_pred", y_pred)

    for ind in range(n_iterations):

        # Compute el pronóstico con los parámetros actuales
        #print( x_poly)
        #print( x)
        y_pred = np.polyval(params, x) #np.___(___, ___) 

        #print("y",type(y),y)
        #print("y_pred",type(y_pred), y_pred )
        #print("y-y_pred",y- y_pred )
        # Calcule el error
        errors = y - y_pred
        #print("errors",errors)
        error =  sum(e**2 for e in errors)


        #print("error",error)

        # Calcule el gradiente
        gradient_w0 = -2 * sum(errors)
        gradient_w1 = -2 * sum(
                [e * x_value for e, x_value in zip(errors, x)]
            )
        gradient_w2 = -2 * sum(
                [e * x_value * x_value for e, x_value in zip(errors, x)]
            )


        gradient = np.array([gradient_w2,gradient_w1,gradient_w0])

        #print ("gradient",gradient)
        #print ("params",params)
        #input()

        # Actualice los parámetros
        params = params - learning_rate * gradient
        #print(ind,params)

    #print("y_pres",y_pred)
    return np.array([params[2],params[1],params[0]])
