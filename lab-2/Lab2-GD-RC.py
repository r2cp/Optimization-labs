
# coding: utf-8

# # Laboratorio 2 - *Gradient Descent*
# 
# Universidad Galileo
# Algoritmos en la Ciencia de Datos
# 
# **Rodrigo Rafael Chang Papa**
# 
# **Carné: 19000625**

# ## Carga de librerías

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ***
# ## Problema 1: **Gradient descent** para el problema cuadrático

# In[2]:


# Funciones para obtener la función QP y su gradiente
def qp(x, Q, c):
    return (0.5*np.matmul(c.T, np.matmul(Q, c)) + np.matmul(c.T, x))

def qp_grad(x, Q, c):
    return (np.matmul(Q,x) + c)


# Si el problema es QP, entonces el método de *step size* exacto es tal que: $$ \alpha_k \triangleq \mathrm{arg}\min_{\alpha \geq 0} f\left(x_k - \alpha_k\nabla f(x_k)\right)
# $$. Como se resolvió en clase: $$ \alpha_k = \frac{\nabla f(x_k)^T \nabla f(x_k)}{\nabla f(x_k)^T Q \nabla f(x_k)} $$

# In[3]:


# Función de gradiente en descenso para QP
def gradientDescentQP(Q, c, x0, step_size=1, alfa=0.01, N=100, eps=1e-6):
    
    # Funciones para evaluar la función y su gradiente
    f = lambda x: qp(x, Q, c)
    grad_f = lambda x : qp_grad(x, Q, c)
    
    # Asignamos el valor inicial
    x = x0
    
    # Listas para guardar valores útiles
    iterList = []
    gradNormList = []
    solutionList = []
    pkList = []
    
    # Para el número de iteraciones
    for k in range(1, N+1):

        # Guardamos los datos de iteraciones
        iterList.append(k)
        gradNormList.append(np.linalg.norm(grad_f(x)))
        solutionList.append(x.reshape(-1))
        pkList.append(-grad_f(x).reshape(-1))
        
        # Revisamos la norma del vector gradiente
        if np.linalg.norm(grad_f(x)) < eps:
            break
            
        # Dependiendo del parámetro step_size, escogemos alfa
        if (step_size == 0):
            # Learning rate es exacto para el problema QP
            lr = (np.matmul(grad_f(x0).T, grad_f(x0)) / np.matmul(grad_f(x0).T, np.matmul(Q, grad_f(x0))))[0,0]
        elif (step_size == 1):
            # Learning rate es constante e igual a alfa dado
            lr = alfa
        else:
            # Learning rate es variable e igual a 1/k para k>0
            lr = 1/k
            
        # Actualizamos el valor de x_k
        x = x - lr*grad_f(x)
    
    # Generamos un DataFrame con los resultados
    data = {'Iteracion':iterList, 'X_k': solutionList, 'GradNorm':gradNormList, 'P_k':pkList}
    gd_df = pd.DataFrame(data)
    return gd_df


# In[4]:


def graficarNormaGrad(gd_df):
    x = gd_df['Iteracion'].values
    y = gd_df['GradNorm'].values
    plt.plot(x, y, 'r-*')
    plt.title('Norma del vector gradiente')
    plt.xlabel('Iteración k')


# ### Pruebas con la primera matriz $Q_1$ y $c_1$

# In[5]:


Q_1 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
c_1 = np.array([1, 0, 1]).reshape(-1, 1)
x_0 = np.array([3, 5, 7]).reshape(-1, 1)


# In[6]:


# Se obtienen resultados del GD para los diferentes tipos de step size
exact_step_size = gradientDescentQP(Q_1, c_1, x_0, step_size=0, N=30)
print("Método de step size exacto")
print(exact_step_size.tail(10))


# In[7]:


const_step_size = gradientDescentQP(Q_1, c_1, x_0, step_size=1, alfa=0.1, N=30)
print("Método de step size constante")
print(const_step_size.tail(10))


# In[8]:


var_step_size = gradientDescentQP(Q_1, c_1, x_0, step_size=2, N=30)
print("Método de step size variable")
print(var_step_size.tail(10))


# In[9]:


k = exact_step_size['Iteracion'].values

plt.figure(figsize=(16,8))
plt.plot(k, exact_step_size['GradNorm'].values, 'r-', label='step size exacto')
plt.plot(k, const_step_size['GradNorm'].values, 'b-*', label='step size constante')
plt.plot(k, var_step_size['GradNorm'].values, 'y--', label='step size 1/k')
plt.xlabel('Iteración k')
plt.ylabel('Norma del gradiente')
plt.legend();


# ### Pruebas con la primera matriz $Q_2$ y $c_2$

# In[15]:


Q_2 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
c_2 = np.array([1, 0, 1]).reshape(-1, 1)
x_0 = np.array([-1, 2, -3]).reshape(-1, 1)


# In[16]:


# Se obtienen resultados del GD para los diferentes tipos de step size
exact_step_size = gradientDescentQP(Q_2, c_2, x_0, step_size=0, N=30)
print("Método de step size exacto")
print(exact_step_size.tail(10))


# In[17]:


const_step_size = gradientDescentQP(Q_2, c_2, x_0, step_size=1, alfa=0.1, N=30)
print("Método de step size constante")
print(const_step_size.tail(10))


# In[18]:


var_step_size = gradientDescentQP(Q_2, c_2, x_0, step_size=2, N=30)
print("Método de step size variable")
print(var_step_size.tail(10))


# In[19]:


k = exact_step_size['Iteracion'].values

plt.figure(figsize=(16,8))
plt.plot(k, exact_step_size['GradNorm'].values, 'r-', label='step size exacto')
plt.plot(k, const_step_size['GradNorm'].values, 'b-*', label='step size constante')
plt.plot(k, var_step_size['GradNorm'].values, 'y--', label='step size 1/k')
plt.xlabel('Iteración k')
plt.ylabel('Norma del gradiente')
plt.legend();


# ## Conclusiones
# Para el primer conjunto de ejercicios, en general, se observa que la mejor elección de $\alpha_k$ para cada iteración es la que se obtiene como resultado del problema de optimización, es decir, el método de selección exacta. Por otra parte, cuando se escoge $\alpha_k = 0.5$, se obtienen resultados muy similares a los obtenidos con el método exacto. En este sentido, esto da lugar a la idea de que para un determinado problema, es posible encontrar un *step size* (o *learning rate*) ideal para el algoritmo de gradiente en descenso.
# 
# Como se observa en el segundo conjunto de ejercicios, la elección del punto inicial del algoritmo de gradiente en descenso puede mejorar el desempeño del algoritmo a pesar de las diferentes elecciones de $\alpha_k$. En este caso, se puede observar que las 3 variantes convergen más o menos en la misma cantidad de iteraciones a los mismos valores de magnitud para el vector gradiente.

# ***
# ## Problema 2: **Rosenbrock's Function**: función banana

# In[44]:


# Definimos la función Rosenbrock y su gradiente

f = lambda x : 100*(x[1] - x[0]**2)**2 + (1-x[0])**2
grad_f = lambda x : np.array([-400*(x[1] - x[0]**2)*(x[0]) - 2*(1-x[0]), 200*(x[1] - x[0]**2)]).reshape(-1, 1)


# In[45]:


x0 = np.array([0, 0.]).reshape(-1, 1)
print('Función en x0: ', f(x0))
print('Gradiente: \n', grad_f(x0))


# In[46]:


# Función de gradiente en descenso más general, para una función y su gradiente
def gradientDescent_fn(f, grad_f, x0, step_size=1, alfa=0.01, N=100, eps=1e-6):
    
    # Asignamos el valor inicial
    x = x0
    
    # Listas para guardar valores útiles
    iterList = []
    gradNormList = []
    solutionList = []
    pkList = []
    
    # Para el número de iteraciones
    for k in range(1, N+1):

        # Guardamos los datos de iteraciones
        iterList.append(k)
        gradNormList.append(np.linalg.norm(grad_f(x)))
        solutionList.append(x.reshape(-1))
        pkList.append(-grad_f(x).reshape(-1))
        
        # Revisamos la norma del vector gradiente
        if np.linalg.norm(grad_f(x)) < eps:
            break
            
        # Dependiendo del parámetro step_size, escogemos alfa
        if (step_size == 1):
            # Learning rate es constante e igual a alfa dado
            lr = alfa
        else:
            # Learning rate es variable e igual a 1/k para k>0
            lr = 1/k
            
        # Actualizamos el valor de x_k
        x = x - lr*grad_f(x)
    
    # Generamos un DataFrame con los resultados
    data = {'Iteracion':iterList, 'X_k': solutionList, 'GradNorm':gradNormList, 'P_k':pkList}
    gd_df = pd.DataFrame(data)
    return gd_df


# In[60]:


# Llamamos a nuestra función de gradiente en descenso con los parámetros especificados
x0 = np.array([0, 0.]).reshape(-1, 1)
gdResult = gradientDescent_fn(f, grad_f, x0, alfa=0.05, N=1000, eps=1e-8)
print(gdResult)


# Como podemos observar, el algoritmo no converge. Esto se debe posiblemente al tamaño del paso $\alpha_k = 0.05$, así como a la escala pobre del problema. Es por esto que la función es utilizada como *benchmark* (marco de referencia) para la prueba de algoritmos de optimización.
# 
# ### Modificación hacia la convergencia
# Después de variar el punto inicial y disminuir el tamaño del paso (*learning rate*), podemos ver que el algoritmo empieza a converger a la solución global, aunque muy lentamente.

# In[62]:


x0 = np.array([0.9, 1.1]).reshape(-1, 1)
gdResult = gradientDescent_fn(f, grad_f, x0, alfa=0.0001, N=1000, eps=1e-8)
print(gdResult)

