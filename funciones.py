import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure 


def plot(img, name, color_legend):
  """Creamos una grafica donde cada banda de la imagen es visualizada separadamente
      de esta manera podemos ver que bandas capturan mejor la aberracion
      img: imagen 
      name:nombre del plot
      color_legend: lista con los nombres de cada subplot"""
  fig = plt.figure(figsize=(12,3))
  fig.suptitle(name, fontsize=16)
  ax = fig.add_subplot(1, 4, 1)
  ax.imshow(img)
  ax.set_xlabel(name,fontsize=14)
  for idx in range(img.shape[2]):
    ax = fig.add_subplot(1, 4, idx+2) 
    ax.imshow(img[:,:,idx]) 
    ax.set_xlabel(color_legend[idx],fontsize=14)
  plt.show()

def plot_scatter(img, name, color, mask = None):
  """Creamos una grafica donde cada banda de la imagen es representada en forma de histograma
      esto nos ayuda a ver la distribucion de los valores
      img: imagen 
      name:nombre del plot
      color: lista con los colores de cada subplot
      mask: opcional se anade si se quiere ver en el histograma solo una parte de la imagen"""
  fig = plt.figure(figsize=(12,3))
  for idx, col in enumerate(color):
    histr = cv2.calcHist([img],[idx],None,[256],[0,256])
    plt.plot(histr, color = col)
    plt.xlim([0,256])
  plt.title(name)
  plt.show()

def moving_w(k, img, mask, funct):
  """Metodo donde dando un kernel (matriz), movimiento 
      y una region se realize la operacion deseada"""
  idx = np.where(mask == 1)
  iter = idx[1].shape[0]
  print("Total de iteraciones de la MV: {}".format(iter))
  #creacion de la ventana
  margen = int(k/2)
  cols = [None,None]
  fils = [None,None]

  for i in range(iter):
    #control de bordes
    if (idx[0][i] - margen) >= 0:
      cols[0] = idx[0][i] - margen
    else:
      cols[0] = 0
    if (idx[0][i] + margen) < img.shape[0]:
      cols[1] = idx[0][i] + margen
    else:
      cols[1] = img.shape[0] -1
    if (idx[1][i] - margen) >= 0:
      fils[0] = idx[1][i] - margen
    else:
      fils[0] = 0
    if (idx[1][i] + margen) < img.shape[1]:
      fils[1] = idx[1][i] + margen
    else:
      fils[1] = img.shape[1] -1
    
    subimage = img[cols[0]:cols[1], fils[0]:fils[1],:]
    ########AQUI se APLICARIA EL FILTRO
    result = funct(subimage)
    ###################################
    if result[1]:
      img[cols[0]:cols[1], fils[0]:fils[1], :] = result[0]
    else:
      img[idx[0][i], idx[1][i], :] = result[0]
  return img

def otsu_thresh(img, band):
  """Codigo para extraer de manera automatica un valor threshold.
  Codigo basado en: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
  """
  bins_num = np.max(img)
  # Obtenemos histograma de la imagen
  hist, bin_edges = np.histogram(img, bins=bins_num)
  # Normalizamos el histograma
  hist = np.divide(hist.ravel(), hist.max())
  # Calculamos el centro de los bins dels histograma
  bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
  # iteramos sobre hist para obtener las probablidades y calculamos sus medias correspondientes
  weight1 = np.cumsum(hist)
  weight2 = np.cumsum(hist[::-1])[::-1]
  mean1 = np.cumsum(hist * bin_mids) / weight1
  mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

  inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
  # Maximize the inter_class_variance function val
  index_of_max_val = np.argmax(inter_class_variance)
  thresh = bin_mids[:-1][index_of_max_val]
  print("Otsu's algorithm implementation thresholding result: ", thresh)
  return thresh

def autocontraste(x,a,b):
    '''Realiza el autocontraste llevando los puntos a y b a 0 y 225, respectivamente'''
    if x<a:
        y=0
    elif x>b:
        y=255
    else:
        y=round((255/(b-a))*(x-a))
    return y

def realce(x,A,B):
    ''' Función para realizar el realce: si A < B = realce sombras y si A > B = realce claros'''
    if A < 0:
        A=0
    if A > 255:
        A=255
    if B < 0:
        B=0
    if B > 255:
        B=255
    if x <= A:
        y = round(B*x/A)
    else:
        y = round(((255-B)*x+(255*(B-A)))/(255-A))
    return y

def logaritmo(x,alfa=0.5):
    c = 223/np.log(1+(np.e**alfa-1)*223)
    y=c*np.log(1+(np.e**alfa-1)*x)
    return round(y)

def exponencial(x,alfa=10):
    x=x/266
    c = 266/((1+alfa)-1)
    y=c*((1+alfa)**x-1)
    return y


def filtro_prueba(img):
  # gamma = 0.9
  # lookUpTable = np.empty((1,256), np.uint8)
  # for i in range(256):
  #     lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
  # res = cv2.LUT(img, lookUpTable)
  # return (res, True)
  #Filtro donde se opera en todos los pixeles de la MV
  h,l,s = cv2.split(img)
  # quitar sturacion
  new_l = l+100
  new_s = s-100
  hsv_new = cv2.merge([h, new_l, new_s])
  return (hsv_new, True)

def filtro_prueba2(img):
  #Filtro donde se usan todos de contexto y solo se cambia el central
  h,s,v = cv2.split(img)
  # quitar sturacion
  sin_val = cv2.multiply(s, 0.6).astype(np.uint8)
  hsv_new = cv2.merge([h,s,sin_val])
  return (np.argmin(img), False)
