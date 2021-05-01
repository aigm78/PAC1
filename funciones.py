import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure 


def plot(img, name, color_legend, fig_scatter, mask = None):
  """Creamos una grafica donde cada banda de la imagen es visualizada separadamente
      de esta manera podemos ver que bandas capturan mejor la aberracion
      img: imagen 
      name:nombre del plot
      color_legend: lista con los nombres de cada subplot
      fig_scatter: scatterplot de las bandas objeto plt
      mask: opcional se anade si se quiere ver en el histograma solo una parte de la imagen"""
  fig = plt.figure(figsize=(25,5))
  fig.suptitle(name + ': RGB Space', fontsize=16)
  ax = fig.add_subplot(1, 5, 1)
  ax.imshow(img)
  ax.set_xlabel(name,fontsize=14)
  for idx in range(img.shape[2]):
    ax = fig.add_subplot(1, 5, idx+2) 
    ax.imshow(img[:,:,idx]) 
    ax.set_xlabel(color_legend[idx],fontsize=14)
  ax = fig.add_subplot(1, 5, 5) 
  for idx, col in enumerate(color):
    histr = cv2.calcHist([img],[idx],mask,[256],[0,256])
    ax.plot(histr, color = col)
    ax.xlim([0,256])
  plt.show()

def plot_scatter(img, name, color):
  """Creamos una grafica donde cada banda de la imagen es representada en forma de histograma
      esto nos ayuda a ver la distribucion de los valores
      img: imagen 
      name:nombre del plot
      color_legend: lista con los colores de cada subplot"""
  fig = plt.figure()
  for idx, col in enumerate(color):
    histr = cv2.calcHist([img],[idx],None,[256],[0,256])
    plt.plot(histr, color = col)
    plt.xlim([0,256])
  plt.title(name)
  return fig

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
