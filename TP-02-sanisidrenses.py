# Trabajo Práctico 2 - sanisidrenses: Guadalupe Cataneo, Máximo Ballestrero y Juana María Leiton.
"""
Un breve resumen de lo contenido:
1: Cargar los datos de Fashion-MNIST
2: Un par de graficos que pusimios en la introduccion del informe
3: Las funciones usadas para el analisis exploratorio, y luego los graficos usados
4: Las funciones usadas para comparar/elegir/ajustar modelos de clasificacion binaria usando KNN y lego los graficos usados
5: Las funciones usadas para comparar/elegir/ajustar modelos de clasificacion multiclase usando Arboles de Decision y los graficos usados
"""



#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import duckdb as dd
import statistics
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import seaborn as sns
import matplotlib.patches as patches
#%% Cargamos los datos:
fmnist = pd.read_csv("Fashion-MNIST.csv")



#%% Me fijo la cantidad de clases y cómo están representadas
consultaSQL =   """
                SELECT DISTINCT label
                FROM fmnist
                ORDER BY label 
                """
dfConsulta = dd.sql(consultaSQL).df()

#%% Las imagenes presentadas en la introduccion
def separar_clase(df):
    prendas = []
    for i in range(0,10):
        prenda1 = df[df["label"] == i].iloc[[0]] 
        prenda2 = df[df["label"] == i].iloc[[5]]
        prenda3 = df[df["label"] == i].iloc[[8]]
        #me quedo solo con los pixeles:
        prenda1=prenda1.iloc[:,1:-1] 
        prenda2=prenda2.iloc[:,1:-1]
        prenda3=prenda3.iloc[:,1:-1]
        #los agrego a la lista como arrays asi dsp es facil graficarlas:
        prendas.append(prenda1.to_numpy())
        prendas.append(prenda2.to_numpy())
        prendas.append(prenda3.to_numpy())
    return prendas




prendas=separar_clase(fmnist)

fig, axes=plt.subplots(3,10,figsize=(23,6))
fig.patch.set_facecolor('lightgray')
for k in range(0,30):
    i=k%3
    j=k//3
    axes[i,j].imshow(prendas[k].reshape((28,28)),cmap="gray")
    if i==0:
        axes[i,j].set_title("Clase "+str(j),fontsize=25)
    axes[i,j].axis("off")


#%%Funciones para desarrollar el análisis exploratorio

#esta funcion devuelve la cantidad de imagenes de cada clase
def frecuencia_de_cada_clase(df):
    frecuencia=[0,0,0,0,0,0,0,0,0,0]
    for i in range (0,10):
        frecuencia[i] = len(df[df['label'] == i])
    return frecuencia


#esta funcion devuelve, por clase las imagenes promedio y mediana, y ademas una matriz
#que tiene la frecuencia de cada combinacion pixel-intensidad, pero esta ultima no la
#terminamos usando
def obtener_datos(df,clase):
    df=dd.sql("""
              SELECT *
              FROM df
              WHERE label="""+str(clase)
              ).df()
    array=df.to_numpy()
    #elimino columna de indices y de clase:
    array=array[:,1:785]
    
    #creo los tres arrays que quiero devolver
    frecuencia=np.zeros((256,784))
    img_promedio=np.zeros(784)
    img_mediana=np.zeros(784)
    for j in range(0,784):
        #aca calculamos la mediana de cada pixel j:
        #agarramos todos los pixeles j
        pixeles=sorted(array[:,j])
        mediana=statistics.median(pixeles)
        img_mediana[j]=mediana
        for i in range(0,array.shape[0]):
            #obtenemos la intensidad del pixel j en la imagen i
            intensidad=array[i,j]
            #vamos sumando todas las intensidades del pixel j
            img_promedio[j]+=intensidad
            #en el array de frecuencias sumamos 1 a la frecuencia que tiene el pixel j de tener intensidad
            frecuencia[intensidad,j]+=1
    #dividimos por 7000 a cada elemento del array de img_promedio asi teniendo en cada elemento el pixel promedio
    img_promedio=img_promedio / 7000
    return frecuencia,img_promedio,img_mediana



#funcion que devuelve la imagen promedio como df, para poder plotear la distribucion de intensidades
#en ella.
def datos_para_graficos(df, clase):
    df=dd.sql("""
              SELECT *
              FROM df
              WHERE label="""+str(clase)
              ).df()
    array=df.to_numpy()
    array=array[:,1:785] #elimino columna de indices y de clase:
    
    #creo los arrays que quiero devolver
    img_promedio=np.zeros(784)
    
    for i in range(0,array.shape[1]): 
        suma = 0
        for j in range(0,array.shape[0]):
            suma += array[j,i]
        img_promedio[i] = suma/7000
        df_nuevo = pd.DataFrame(img_promedio, columns= ['Promedio pixeles'])
        df_nuevo['label'] = clase
    return df_nuevo


# Funciones que calculan dispersion por pixel: la primera por clase, y la segunda comparando entre clases

#dada una clase, le calcula a cada pixel su iqr
#(la idea es asi ver cuantos y que tanto varian las intensidades de los pixeles de una clase)
def calcular_iqr_por_pixel_clase(df,clase):
    df=dd.sql("""
              SELECT *
              FROM df
              WHERE label="""+str(clase)
              ).df()
    array=df.to_numpy()
    array=array[:,1:785]
    iqrs=np.zeros(784)
    for j in range(0,784):
        pixeles=sorted(array[:,j])
        iqrs[j]= np.percentile(pixeles, 75) - np.percentile(pixeles, 25)
    return iqrs

#calcula primero las imagenes medianas de cada clase, y luego a partir de ellas, a cada pixel le calcula
#el rango (la idea es asi tener una idea de los pixeles que mas varian clase a clase)
def calcular_rango_por_pixel(df):
    imgs_medi=[]
    res=[]
    for i in range(0,10):
        imgs_medi.append( obtener_datos(df, i)[2] )
    imgs_medi=np.array(imgs_medi)
    for j in range(0,784):
        pixel_j=imgs_medi[:,j]
        rango_j=    np.max(pixel_j)-np.min(pixel_j) #np.percentile(pixel_j,75)-np.percentile(pixel_j,25)
        res.append(rango_j)
    return np.array(res)
#%% vemos frecuencias de cada clase
frecuencias=frecuencia_de_cada_clase(fmnist)
print(frecuencias)

#%% Grafico de imagenes promedio y mediana de todas las clases.
datos=[]
for i in range(0,10):
    datos.append(obtener_datos(fmnist,i))


fig,axes=plt.subplots(2,10,figsize=(20,4))
fig.patch.set_facecolor('lightgray')
for i in range(0,2):
    for j in range(0,10):
        axes[i,j].imshow(datos[j][i+1].reshape(28,28), cmap='gray')
        if i==0:
            axes[i,j].set_title("Clase "+str(j),fontsize=25)
        axes[i,j].axis("off")
axes[0,0].text(-7, 14, "Promedio", fontsize=16, rotation=90, va='center')
axes[1,0].text(-7, 14, "Mediana", fontsize=16, rotation=90, va='center')


plt.show()
plt.close()   



#%% Separo información para comparar clases 1, 2 y 6
prom_clase_1 = datos_para_graficos(fmnist, 1)
prom_clase_2 = datos_para_graficos(fmnist, 2)
prom_clase_6 = datos_para_graficos(fmnist, 6)

#%% Histogramas
fig, axs=plt.subplots(1,3,figsize=(18,5), facecolor='lightgrey')
fig.patch.set_facecolor('lightgray')
# clase 1
axs[0].hist(prom_clase_1['Promedio pixeles'], bins=50, color='#0173b2',edgecolor='black')
axs[0].set_title('Clase 1', fontsize=20)
axs[0].set_xlabel('Intensidades', fontsize=15)
axs[0].set_ylabel('Cantidad de pixeles', fontsize=15)
axs[0].set_ylim(0,350)
axs[0].set_xlim(0,220)
# clase 2
axs[1].hist(prom_clase_2['Promedio pixeles'], bins=50, color='#de8f05',edgecolor='black')
axs[1].set_title('Clase 2', fontsize=20)
axs[1].set_xlabel('Intensidades', fontsize=15)
axs[1].set_ylabel('Cantidad de pixeles', fontsize=15)
axs[1].set_ylim(0,350)
axs[1].set_xlim(0,220)
#clase 6
axs[2].hist(prom_clase_6['Promedio pixeles'], bins=50, color='#029e73',edgecolor='black')
axs[2].set_title('Clase 6', fontsize=20)
axs[2].set_xlabel('Intensidades', fontsize=15)
axs[2].set_ylabel('Cantidad de pixeles', fontsize=15)
axs[2].set_ylim(0,350)
axs[2].set_xlim(0,220)

plt.tight_layout()
plt.show()

#%% Histogramas con zoom
fig, axs=plt.subplots(1,3,figsize=(18,5), facecolor='lightgrey')
fig.patch.set_facecolor('lightgray')
# clase 1
axs[0].hist(prom_clase_1['Promedio pixeles'], bins=50, color='#0173b2',edgecolor='black')
axs[0].set_title('Clase 1', fontsize=20)
axs[0].set_xlabel('Intensidades', fontsize=15)
axs[0].set_ylabel('Cantidad de pixeles', fontsize=15)
axs[0].set_ylim(0,150)
axs[0].set_xlim(0,220)
axs[0].tick_params(axis='both', labelsize=15)
# clase 2
axs[1].hist(prom_clase_2['Promedio pixeles'], bins=50, color='#de8f05',edgecolor='black')
axs[1].set_title('Clase 2', fontsize=20)
axs[1].set_xlabel('Intensidades', fontsize=15)
axs[1].set_ylabel('Cantidad de pixeles', fontsize=15)
axs[1].set_ylim(0,150)
axs[1].set_xlim(0,220)
axs[1].tick_params(axis='both', labelsize=15)
#clase 6
axs[2].hist(prom_clase_6['Promedio pixeles'], bins=50, color='#029e73',edgecolor='black')
axs[2].set_title('Clase 6', fontsize=20)
axs[2].set_xlabel('Intensidades', fontsize=15)
axs[2].set_ylabel('Cantidad de pixeles', fontsize=15)
axs[2].set_ylim(0,150)
axs[2].set_xlim(0,220)
axs[2].tick_params(axis='both', labelsize=15)

plt.tight_layout()
plt.show()

#%% Boxplot y violinlot para comparar clases 1, 2 y 6

# Concateno verticalmente las columnas de los dataframes con los promedios de los valores para cada pixel de clase 1 y 2
col1 = pd.concat([prom_clase_1['Promedio pixeles'], prom_clase_2['Promedio pixeles'], prom_clase_6['Promedio pixeles']], ignore_index=True)
col2 = pd.concat([prom_clase_1['label'], prom_clase_2['label'], prom_clase_6['label']], ignore_index=True)

# Armo nuevo dataframe
prom_clase_1_vs_2_vs_6 = pd.DataFrame({
    'Promedio pixeles': col1,
    'label': col2})

pixel_cols = [col for col in fmnist.columns if col.startswith('pixel')]
# Creamos el nuevo DataFrame
tamanios_con_clases = pd.DataFrame()
tamanios_con_clases['tamanio'] = (fmnist[pixel_cols] != 0).sum(axis=1)
tamanios_con_clases['label'] = fmnist['label']

pixel_cols = [col for col in fmnist.columns if col.startswith('pixel')]
# Creamos el nuevo DataFrame
tamanios_con_clases = pd.DataFrame()
tamanios_con_clases['tamanio'] = (fmnist[pixel_cols] != 0).sum(axis=1)
tamanios_con_clases['label'] = fmnist['label']

#hacemos el boxplot de distribucion de intensidad en img promedio
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='lightgrey')
sns.boxplot(data=prom_clase_1_vs_2_vs_6, x='label', y='Promedio pixeles',
            palette='colorblind', width=0.4, fliersize=5, ax=axes[0])
promedio = prom_clase_1_vs_2_vs_6.groupby('label')['Promedio pixeles'].mean()
for i, (label, prom) in enumerate(promedio.items()):
    axes[0].plot(i, prom, marker='^', color='black', markersize=8, label='Promedio' if i == 0 else "")
axes[0].set_title('Fig. 4')
axes[0].set_xlabel('Clases', fontsize=14)
axes[0].set_ylabel('Promedio pixeles', fontsize=14)
axes[0].grid(True, linestyle='--', alpha=0.5)

#hacemos el violin plot de tamaños
datos = tamanios_con_clases[tamanios_con_clases['label'].isin([1, 2, 6])]
sns.violinplot(x="label", y="tamanio", data=datos, palette='colorblind', ax=axes[1])
axes[1].set_title('Fig. 5')
axes[1].set_xlabel('Clases', fontsize=14)
axes[1].set_ylabel('Tamaño', fontsize=14)
axes[1].set_ylim(0, None)
plt.tight_layout()
plt.show()


#%% Grafico para clases 1 y 8, el iqr de cada pixel

iqrs1=calcular_iqr_por_pixel_clase(fmnist, 1)
iqrs8=calcular_iqr_por_pixel_clase(fmnist, 8)

vmin=0 
vmax=max( [max(iqrs1),max(iqrs8)] )
fig,axes=plt.subplots(1,2,figsize=(9.5,4))

axes[0].imshow(iqrs1.reshape((28,28)), cmap="viridis", vmin=vmin, vmax=vmax)
axes[0].set_title("Clase 1")
axes[0].axis('off')
img=axes[1].imshow(iqrs8.reshape((28,28)), cmap="viridis", vmin=vmin, vmax=vmax)
axes[1].set_title("Clase 8")
axes[1].axis('off')

cbar=fig.colorbar(img,label="IQR")
plt.tight_layout()
plt.show()

#%% Heatmap rango de píxeles
a=calcular_rango_por_pixel(fmnist)

#grafico el heatmap:
im=a.reshape((28,28))
plt.imshow(im,cmap="viridis")
plt.colorbar(label="Rango")
plt.axis('off')
plt.show()
plt.close()








#%%Análisis clasificación binaria:
#%% Funciones para la clasificacion binaria con KNN:
def n_pixeles_max_abs(array,n):
    array_abs=np.abs(array)
    nombres_pixeles_relevantes=[]
    for i in range (0,n):
        id_max=np.argmax(array_abs)
        array_abs[id_max]=0 
        nombres_pixeles_relevantes.append("pixel"+str(id_max))
    return nombres_pixeles_relevantes


#graficos de selecciones de pixeles
def graficar_selecciones_de_pixeles(selecciones,cantidades,img):
    fig, axes=plt.subplots(1,3,figsize=(10,3))
    fig.patch.set_facecolor('lightgray')

    for i in range(0,3):
        axes[i].imshow(img,cmap='bwr',vmin=-255,vmax=255)
        axes[i].axis('off')
        axes[i].set_title(cantidades[i]+' pixeles')
        pixeles=selecciones[i]
        for j in range(0,len(pixeles)):
            pixel=int(pixeles[j][5:])
            fila = pixel // 28
            col = pixel % 28
            rect=patches.Rectangle((col - 0.5, fila -0.5), 1, 1, linewidth=1, edgecolor='lime', facecolor='none')
            axes[i].add_patch(rect)
    plt.show()

# Dado una lista de ks, te grafica sus accuracies, y 
# te devuelve el k tal que la accuracy es maxima, y esa accuracy
def plot_accuracy(ks,X_eval,y_eval,pixeles,X_dev,y_dev):
    #X=fmnist_dev_0_8[pixeles]
    #y=fmnist_dev_0_8['label']
    accuracies_train=[]
    accuracies_test=[]
    for k in ks:
        clf=KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_dev[pixeles],y_dev)
        y_pred_test=clf.predict(X_eval[pixeles])
        y_pred_train=clf.predict(X_dev[pixeles])
        accuracy_test=accuracy_score(y_eval,y_pred_test)
        accuracy_train=accuracy_score(y_dev,y_pred_train)
        accuracies_test.append(accuracy_test)
        accuracies_train.append(accuracy_train)
    fig, ax = plt.subplots() 
    plt.plot(ks,accuracies_test, marker='.', color='skyblue',label='test')
    plt.plot(ks,accuracies_train, marker='.', color='orange',label='train')
    ax.set_xlabel('Valores de k')
    ax.set_ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.close()
    maximo_test=(np.argmax(accuracies_test),max(accuracies_test))
    return  "Máximo k test: " + str(ks[maximo_test[0]]) +";   Accuracy max: " + str(maximo_test[1]) 

# lo mismo pero subplots:    
def plot_accuracy_subplots(ax,ks,X_eval,y_eval,pixeles,X_dev,y_dev,nombre):
    accuracies_train=[]
    accuracies_test=[]
    for k in ks:
        clf=KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_dev[pixeles],y_dev)
        y_pred_test=clf.predict(X_eval[pixeles])
        y_pred_train=clf.predict(X_dev[pixeles])
        accuracy_test=accuracy_score(y_eval,y_pred_test)
        accuracy_train=accuracy_score(y_dev,y_pred_train)
        accuracies_test.append(accuracy_test)
        accuracies_train.append(accuracy_train)
    ax.plot(ks,accuracies_test, marker='.', color='skyblue',label='test')
    ax.plot(ks,accuracies_train, marker='.', color='orange',label='train')
    ax.set_ylim(0.88, 1.0)
    ax.set_xlabel('Valores de k',fontsize=15)
    ax.set_ylabel('Accuracy',fontsize=15)
    ax.set_title(nombre,fontsize=20)
    ax.legend()
    
    maximo_test=(np.argmax(accuracies_test),max(accuracies_test))
    return  "Máximo k test: " + str(maximo_test) 

#%% Obtenemos el sub-datataframe donde las clases son unicamente 0 y 8
fmnist_clases_0_8 = fmnist[fmnist['label'].isin([0,8])]
muestras_clase_0 = len(fmnist[fmnist['label'] == 0]) #contamos cuantas muestras hay de la clase 0
muestras_clase_8 = len(fmnist[fmnist['label'] == 8]) #contamos cuantas muestras hay de la clase 8
#%% Separamos este sub-dataframe en train y test:
X = fmnist_clases_0_8.drop('label',axis=1)
y = fmnist_clases_0_8.label
X_dev, X_eval, y_dev, y_eval = train_test_split(X,y,test_size=0.2, random_state=20)
y_dev_df=y_dev.to_frame(name="label")

fmnist_dev_0_8=X_dev.join(y_dev_df,how="left")
#%% Grafico restas de imagenes promedio y mediana entre clases 0 y 8
dif_img_promedio_8_0=obtener_datos(fmnist_dev_0_8,8)[1]-obtener_datos(fmnist_dev_0_8,0)[1]
img_resta_prom = np.array(dif_img_promedio_8_0.reshape((28, 28)))

dif_img_mediana_8_0=obtener_datos(fmnist_dev_0_8,8)[2]-obtener_datos(fmnist_dev_0_8,0)[2]
img_resta_medi = np.array(dif_img_mediana_8_0.reshape((28, 28)))

fig, axes=plt.subplots(1,2,figsize=(9.5,4))
fig.patch.set_facecolor('lightgray')
axes[0].imshow(img_resta_prom,cmap="bwr",vmin=-255,vmax=255)
axes[0].set_title("Diferencia promedios")
axes[0].axis('off')
im=axes[1].imshow(img_resta_medi,cmap="bwr",vmin=-255,vmax=255)
axes[1].set_title("Diferencia medianas")
axes[1].axis('off')

cbar=fig.colorbar(im,label="Diferencia")
plt.tight_layout()
plt.show()

#%%
#3 PIXELES:
tres_pix_prom=n_pixeles_max_abs(dif_img_promedio_8_0,3) 
tres_pix_medi=n_pixeles_max_abs(dif_img_mediana_8_0,3)
#consideramos una mezcla de ambos, eligiendo para que cada pixel sea de una seccion distinta de la imagen
tres_pix_elegidos=[tres_pix_prom[0],tres_pix_prom[2],tres_pix_medi[0]]

#5 PIXELES:
cinco_pix_prom = n_pixeles_max_abs(dif_img_promedio_8_0,5)
cinco_pix_medi = n_pixeles_max_abs(dif_img_mediana_8_0,5)
# elegimos los dos de la parte superior mas distanciados (medi), uno central inferior (medi), uno de cada lado (nubes rojas, prom)
cinco_pix_elegidos = [cinco_pix_medi[0], cinco_pix_medi[2], cinco_pix_prom[0], cinco_pix_prom[2], cinco_pix_medi[3]]

#10 PIXELES:
diez_pix_prom = n_pixeles_max_abs(dif_img_promedio_8_0, 10)
diez_pix_medi = n_pixeles_max_abs(dif_img_mediana_8_0, 10)
#consideramos una mezcla de ambos, eligiendo para que cada pixel sea de una seccion distinta de la imagen
diez_pix_elegidos = diez_pix_prom[0:5] + diez_pix_medi[0:5]


#%% Gráficos para el anexo
selecciones_prom=[tres_pix_prom, cinco_pix_prom, diez_pix_prom]
selecciones_medi=[tres_pix_medi, cinco_pix_medi, diez_pix_medi]
selecciones_elegidos=[tres_pix_elegidos, cinco_pix_elegidos, diez_pix_elegidos]
cantidades=["3","5","10"]
graficar_selecciones_de_pixeles(selecciones_prom, cantidades,img_resta_prom)
graficar_selecciones_de_pixeles(selecciones_medi, cantidades,img_resta_medi)
graficar_selecciones_de_pixeles(selecciones_elegidos, cantidades,img_resta_medi)



#%%
lista_ks=[1,2,3,4,5,6,7,8,9,10,20,30,40,50,75,100,125,150,175,200]
#%% clasificador 1: promedio
plot_accuracy(lista_ks,X_eval,y_eval,tres_pix_prom,X_dev,y_dev) #accuracy max: 0.9210, k=40
plot_accuracy(lista_ks,X_eval,y_eval,cinco_pix_prom,X_dev,y_dev) #accuracy max: 0.9275, k=20
plot_accuracy(lista_ks,X_eval,y_eval,diez_pix_prom,X_dev,y_dev) #accuracy max: 0.9321, k=40
#%% clasificador 2: mediana
plot_accuracy(lista_ks,X_eval,y_eval,tres_pix_medi,X_dev,y_dev) #accuracy max: 0.8689, k=20
plot_accuracy(lista_ks,X_eval,y_eval,cinco_pix_medi,X_dev,y_dev) #accuracy max: 0.8746, k=7
plot_accuracy(lista_ks,X_eval,y_eval,diez_pix_medi,X_dev,y_dev) #accuracy max: 0.9314, k=75
#%% clasificador 3: elegido
plot_accuracy(lista_ks,X_eval,y_eval,tres_pix_elegidos,X_dev,y_dev) #accuracy max: 0.9257, k=20
plot_accuracy(lista_ks,X_eval,y_eval,cinco_pix_elegidos,X_dev,y_dev) #accuracy max: 0.9367, k=30
plot_accuracy(lista_ks,X_eval,y_eval,diez_pix_elegidos,X_dev,y_dev) # accuracy max: 0.9446, k=6

#despues de los tres criterios para tomar conjuntos de pixeles
#concluimos que el mejor criterio (el que para la lista de ks alcanza
#siempre accuracy mas alta, para las 3 cantidades de pixeles) 
#es el tercero, asi que en la proxima
#celda graficamos su accuracy en funcion de K, para conjunto train y test.

#ademas concluimos que el mejor clasificador de todos es el de diez
#pixeles, con este tercer criterio y k=6, asi que graficaremos su matriz
#de confusion mas adelante tambien..

#%%
#aca un subplot con las accuracies en funcion de K para los 3 conjuntos elegidos con el tercer criterio    
fig,axs = plt.subplots(1,3,figsize=(18,5))    
tres_pixeles = plot_accuracy_subplots(axs[0],lista_ks,X_eval,y_eval,tres_pix_elegidos,X_dev,y_dev,"3 píxeles")
cinco_pixeles = plot_accuracy_subplots(axs[1],lista_ks,X_eval,y_eval,cinco_pix_elegidos,X_dev,y_dev,"5 píxeles")
diez_pixeles = plot_accuracy_subplots(axs[2],lista_ks,X_eval,y_eval,diez_pix_elegidos,X_dev,y_dev,"10 píxeles")
fig.patch.set_facecolor('lightgrey')
plt.tight_layout()
plt.show()
#%% matriz de confusion para el mejor clasificador binario (10 pixeles de los elegidos, k=6)

mejor_clf_binario=KNeighborsClassifier(n_neighbors=6)
mejor_clf_binario.fit(X_dev[diez_pix_elegidos],y_dev)
y_pred=mejor_clf_binario.predict(X_eval[diez_pix_elegidos])
m_de_confusion=confusion_matrix(y_eval,y_pred)


#plt.figure(figsize=(,3))
sns.heatmap(m_de_confusion, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=[0,8], yticklabels=[0,8])
plt.title("Matriz de confusión")
plt.xlabel("Clase predicha")
plt.ylabel("Clase verdadera")
plt.tight_layout()
plt.show()











#%%Analisis de clasificacion multiclase:
#%%
# hiperparámetros árbol
# (2)criterio: gini y entropía
# (10)max_depth: de 1 a 10
# random_state: semilla cualquiera para que de siempre el mismo modelo

#%%Funciones:
# función que ajusta y calcula accuracy de árboles que varían sus profundidades de 1 a 10, luego plotea los resultados
def plot_accuracy_arboles(X_train, y_train, X_test, y_test):
    alturas_de_arbol = []
    accuracies = []
    for i in range(1,11):
        alturas_de_arbol.append(i)
        arbol = DecisionTreeClassifier(max_depth=i, random_state=5)
        arbol.fit(X_train, y_train)
        y_pred = arbol.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    plt.figure(figsize=(8, 5))
    plt.plot(alturas_de_arbol, accuracies, marker='o', linestyle='-', color='#0173b2', label='Accuracy')
    plt.title('Accuracy según cantidad de profundidades')
    plt.xlabel('Cantidad de profundidades')
    plt.ylabel('Accuracy')
    plt.xticks(alturas_de_arbol)  # para que se vean todos los valores de k en el eje x
    plt.legend()
    plt.tight_layout()
    plt.show()

# función que devuelve el accuracy + configuración de hiperparámetros del mejor árbol de los posibles de ajustar con las 
# combinaciones de los hiperparámetros que se le pasen (usa k-folding para validación cruzada)
def comparar_arboles_kfolding(h,X_desarrollo, y_desarrollo):
    res = []
    cantidad_arboles = len(h[0])*len(h[1])
    hp = []
    for crit in h[0]:
        for maxdepth in h[1]:
            hp.append([crit,maxdepth])
    # hacemos validación cruzada con k-folding usando 5 folds
    kf = KFold(n_splits=5)
    # vamos almacenando la exactitud en cada fold por cada modelo
    resultados = np.zeros((5,cantidad_arboles))
    for i, (train_index, test_index) in enumerate(kf.split(X_desarrollo)):
        kf_X_train, kf_X_test = X_desarrollo.iloc[train_index], X_desarrollo.iloc[test_index]
        kf_y_train, kf_y_test = y_desarrollo.iloc[train_index], y_desarrollo.iloc[test_index]
        for j in range(0,cantidad_arboles):  
            arbol = DecisionTreeClassifier(criterion=hp[j][0],max_depth=hp[j][1],random_state=5)
            arbol.fit(kf_X_train, kf_y_train)
            pred = arbol.predict(kf_X_test)
            score = accuracy_score(kf_y_test,pred)
            resultados[i, j] = score
    # calculamos el promedio de las exactitudes de cada fold por cada modelo
    res = resultados.mean(axis = 0)
    max_i_res = res.argmax()
    mejores_hp = hp[max_i_res]
    res = res[max_i_res]
    return [res, mejores_hp]
#%% Obtenemos los datos
X = fmnist.drop('label',axis=1)
X = X.drop('Unnamed: 0', axis=1)
y = fmnist.label
# Separamos held-out 10% y desarrollo 90%
X_desarrollo, X_eval, y_desarrollo, y_eval = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y)
# Separamos dentro de desarrollo a train 80% y test 20%
X_train, X_test, y_train, y_test = train_test_split(
    X_desarrollo, y_desarrollo, test_size=0.2, random_state=7, stratify=y_desarrollo)

#%% grafiquemos las accuracies de los árboles con profundidades de 1 a 10, usando solo train y test, sin K-folding ni variando criterio
plot_accuracy_arboles(X_train, y_train, X_test, y_test)

#%% comparamos los distintos arboles posibles con K-folding
comparar_arboles_kfolding([['gini','entropy'],[1,5,10]],X_desarrollo, y_desarrollo)
# vemos que el mejor es entropía con 10, ajustamos un poco mas el parametro de max_depth
#%% reajustamos:
mejor_arbol_datos = comparar_arboles_kfolding([['gini','entropy'],[8,9,10]],X_desarrollo, y_desarrollo)
# vemos que el mejor es entropía con 10, con una exactitud promedio de 0.8092
#%%
# ahora entrenamos al árbol seleccionado con todo el conjunto de datos de desarrollo y lo testeamos con el conjunto held-out que hasta ahora no tocamos
index_mejor_arbol = mejor_arbol_datos[0]
mejores_hp = mejor_arbol_datos[1]
mejor_arbol = DecisionTreeClassifier(criterion=mejores_hp[0], max_depth=mejores_hp[1], random_state=5)
mejor_arbol.fit(X_desarrollo,y_desarrollo)
y_pred = mejor_arbol.predict(X_eval)
accuracy = accuracy_score(y_eval,y_pred)
# observamos que la accuracy del árbol luego de testearlo con held-out fue de 0.8133
# hacemos la matriz de confusion
matriz_de_confusion = confusion_matrix(y_eval,y_pred)
# ahora analizamos más específicamente la exactitud de cada clase por separado 
exactitud_por_clase = []
for clase in range (0,10):
    exactitud_por_clase.append([clase,(matriz_de_confusion[clase][clase])*10/matriz_de_confusion[clase].sum()])
# podemos ver que hay algunas clases con exactitud más baja al resto
#%%
# graficamos el arbol de decision
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('lightgray')
tree.plot_tree(mejor_arbol, max_depth=1, proportion=True, ax=ax, fontsize=10)
#ax.set_facecolor("lightgray")
plt.show()
#%%
# gráfico de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_de_confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Matriz de confusión")
plt.xlabel("Clase predicha")
plt.ylabel("Clase verdadera")
plt.tight_layout()
plt.show()
#%%