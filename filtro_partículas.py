import time

import cv2
import os
import numpy as np
import random

def calcular_pesos(img, particulas):
    h, w = 30, 30
    estados = particulas['estados']

    for i in range(estados.shape[0]):
        y = estados[i, 0]
        x = estados[i, 1]
        particulas['peso'][i] = np.sum(img[y:y+h, x:x+w])

    total = np.sum(particulas['peso'])

    if total != 0:
        particulas['peso'] = particulas['peso']/total

def select_best_particle(particulas):
    index = np.argmax(particulas['peso'])
    if particulas['peso'][index] > 0:
        return particulas['estados'][index,0], particulas['estados'][index,1]
    else:
        return None, None


def select_particles(particulas):

    y_best, x_best = select_best_particle(particulas)

    pesos_acum = particulas['peso']

    # Calcular CDF
    for i in range(1, pesos_acum.shape[0]):
        pesos_acum[i] = pesos_acum[i-1] + pesos_acum[i]

    random.seed(None)

    # Remuestrear de la distribución
    for i in range(pesos_acum.shape[0]):
        rand = random.random()
        j = 0

        # Seleccionar primer ejemplo con probabilidad acumulada mayor que rand
        while j < pesos_acum.shape[0] and pesos_acum[j] < rand:
            j = j+1

        # Muestrear de una normal con media en el estado j.
        if j < pesos_acum.shape[0]:
            mu, sigma = 0, 20  # mean and standard deviation

            particulas['estados'][i,0] = particulas['estados'][j,0] + int(np.random.normal(mu, sigma))
            particulas['estados'][i,1] = particulas['estados'][j,1] + int(np.random.normal(mu, sigma))

    return y_best, x_best


def print_particles(img, particulas, y_best = None, x_best = None):
    h, w = 30, 30

    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if particulas:
        index = np.random.choice(range(len(particulas['estados'])), int(len(particulas['estados'])*0.5), replace=False)
        estados = particulas['estados'][index]
        for i in range(estados.shape[0]):
            y = estados[i, 0]
            x = estados[i, 1]
            cv2.circle(img, center=(int(x+w/2), int(y+h/2)), radius=3, color=(0, 255, 0), thickness=-1)
            # cv2.rectangle(img, (x,y),(x+w,y+h), (0, 255, 0), 1)

    if y_best and x_best:
        cv2.rectangle(img, (x_best, y_best), (x_best + w, y_best + h), (0, 0, 255), 2)

    cv2.imshow('frame', img)
    cv2.waitKey(1)



path = r'SecuenciaPelota'
img_paths = [os.path.join(path, '{}.jpg'.format(i)) for i in range(1, 61)]

background = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE)

img_h, img_w = background.shape
n = 200
particulas = {'estados': np.zeros((n,2)),
              'peso': np.zeros(n)}

y_best, x_best = None, None

for img_path in img_paths:
    frame_rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    # frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if not y_best and not x_best:
        particulas['estados'][:, 0] = np.random.rand(n) * img_h
        particulas['estados'][:, 1] = np.random.rand(n) * img_w
        particulas['estados'] = particulas['estados'].astype(int)

    # Sustraccion de fondo
    th = np.abs(np.float32(frame) - np.float32(background)) > 70
    th = np.uint8(th * 255)

    #calculo pesos
    calcular_pesos(th, particulas)

    # print_particles(frame_rgb, particulas)

    y_best, x_best = select_particles(particulas)

    print_particles(frame_rgb, particulas, y_best, x_best)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    time.sleep(0.2)
cv2.destroyAllWindows()