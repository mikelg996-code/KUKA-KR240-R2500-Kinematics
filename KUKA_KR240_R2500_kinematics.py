# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:22:10 2024

@author: MIKEL
"""

import numpy as np


'''
*******************************************************************************
                        PARAMETROS DE ENTRADA
*******************************************************************************
'''

'************************   BASE   ************************'
'B[0]'
BASE = np.zeros(6) 
'BASE - prueba'
# BASE = np.array([
#         -1124.862,  # X (mm)
#         -1872.222,  # Y (mm)
#         -662.488,   # Z (mm)
#         1.321,      # A (grados)
#         -0.016,     # B (grados)
#         -0.135      # C (grados)
#         ])

'************************   TOOL   ************************'
'T[0]'
TOOL = np.zeros(6) 
'T - Neumatico compensado'
# TOOL = np.array([
#         311.178,       # X (mm)
#         0.002,         # Y (mm)
#         334,           # Z (mm)
#         -180,          # A (grados)
#         -90,           # B (grados)
#         0              # C (grados)
#         ])

'****************   Denavit-Hartenberg   ******************'
'NOMINALES'
dh_nominales =  np.array([
    (-90,     350,      0,      675),  # J1 [alpha, a, theta, d]
    (  0,    1150,    -90,        0),  # J2
    (-90,     -41,      0,        0),  # J3
    ( 90,       0,      0,     1000),  # J4
    (-90,       0,    180,        0),  # J5
    (  0,       0,      0,      240),  # J6
])

invertir_base    = False
invertir_codo    = False
invertir_muneca  = False

'****************   OPW parametros   ******************'
'NOMINAL'
OPW_nominales = np.array([
    350,    # a1 [mm]
    41,     # a2 [mm] (J3 por encima de J4)
    675,    # c1 [mm]
    1150,   # c2 [mm]
    1000,   # c3 [mm]
    240,    # c4 [mm]
    0       # d  [mm]
    ])

'Definir los límites para cada articulación'
limits = {
    'joint1': (-185, 185),
    'joint2': (-140, -5),
    'joint3': (-120, 155),
    'joint4': (-179, 179),
    'joint5': (-122.5, 122.5),
    'joint6': (-179, 179),
}

'''
*******************************************************************************
                        FUNCIONES
*******************************************************************************
'''
def denavit(theta, d, a, alpha):
    """Función para calcular la matriz de transformación de Denavit-Hartenberg.
        (RADIANES)
    """
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0,              np.sin(alpha),                 np.cos(alpha),                d],
        [0,              0,                              0,                            1]
    ])

def extract_euler_angles(T):
    """
    Extrae los ángulos de Euler (A, B, C) a partir de una matriz de transformación homogénea T
    bajo la convención ZYX (giro alrededor de Z, luego Y, luego X).
    """
    # Extraer componentes de la matriz de rotación
    r11, r12, r13 = T[0, :3]
    r21, r22, r23 = T[1, :3]
    r31, r32, r33 = T[2, :3]
    
    # Calcular sy para verificar singularidad
    sy = np.sqrt(r11**2 + r21**2)
    singular = sy < 1e-6

    if not singular:
        # Si no es singular, usar las ecuaciones estándar para ZYX
        A = np.arctan2(r21, r11)  # Rotación alrededor de Z
        B = np.arctan2(-r31, sy)  # Rotación alrededor de Y
        C = np.arctan2(r32, r33)  # Rotación alrededor de X
    else:
        # Caso singular (gimbal lock)
        A = np.arctan2(-r12, r22)
        B = np.arctan2(-r31, sy)
        C = 0  # No se puede determinar C en este caso

    # Convertir a grados
    A_deg = np.degrees(A)
    B_deg = np.degrees(B)
    C_deg = np.degrees(C)
    
    return A_deg, B_deg, C_deg

def rot_z(angle_deg):
    angle_rad = np.radians(angle_deg)
    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0, 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def rot_x(angle_deg):
    angle_rad = np.radians(angle_deg)
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad), 0],
        [0, np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0, 0, 0, 1]
    ])

def rot_y(angle_deg):
    angle_rad = np.radians(angle_deg)
    return np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad), 0],
        [0, 1, 0, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
        [0, 0, 0, 1]
    ])

def translation(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

def transformation_matrix(x, y, z, a, b, c):
    """Crea una matriz de transformación homogénea a partir de una traslación (x, y, z)
    y una rotación de Euler (a, b, c) en grados (A - Z, B - Y, C - X)"""
    T = translation(x, y, z)
    R = rot_z(a) @ rot_y(b) @ rot_x(c)
    return T @ R

def direct_kinematic(joint_angles,dh_robot):
    
    # global A01, theta, alfa
    """Direct Kinematic.
    
    Calcula la cinematica directa
    """
    # Parámetros Denavit-Hartenberg del robot   
    alfa = np.radians(dh_robot[:,0]) # en radianes
    a = dh_robot[:,1]
    d = dh_robot[:,3]
    #--------------------------------------------------------------------------
    theta = [
        dh_robot[0,2]-joint_angles[0],         # theta1
        dh_robot[1,2]+joint_angles[1]+90,      # theta2
        dh_robot[2,2]+joint_angles[2]-90,      # theta3
        dh_robot[3,2]-joint_angles[3],         # theta4
        dh_robot[4,2]+joint_angles[4]+180,     # theta5
        dh_robot[5,2]-joint_angles[5]+180,     # theta6
        ]
    theta = np.radians(theta) # en radianes
    
    # Matrices de transformación homogénea entre sistemas de coordenadas consecutivos
    A01 = denavit(theta[0], d[0], a[0], alfa[0])
    A12 = denavit(theta[1], d[1], a[1], alfa[1])
    A23 = denavit(theta[2], d[2], a[2], alfa[2])
    A34 = denavit(theta[3], d[3], a[3], alfa[3])
    A45 = denavit(theta[4], d[4], a[4], alfa[4])
    A56 = denavit(theta[5], d[5], a[5], alfa[5])

    # Matriz de transformación del primer al último sistema de coordenadas
    A06 = A01 @ A12 @ A23 @ A34 @ A45 @ A56  # Producto matricial
    
    'extraer valores coordenadas'
    # X = A06[0,3]
    # Y = A06[1,3]
    # Z = A06[2,3]
    # A, B, C = extract_euler_angles(A06)
    
    return A06


def inverse_kinematic(coordinates,OPW_params,limites):
    
    'Parametros OPW del robot'
    a1 = OPW_params[0]
    a2 = OPW_params[1]
    c1 = OPW_params[2]
    c2 = OPW_params[3]
    c3 = OPW_params[4]
    c4 = OPW_params[5]
    b  = OPW_params[6]
    # .........................................................................
    T_coordinates = transformation_matrix(*coordinates)
    'Calculamos la posición del centro de la muñeca (C)'
    u_e = T_coordinates[:3, 3]  # Posición del efector final
    z_e = T_coordinates[:3, 2]  # Eje z del efector
    c = u_e - c4 * z_e  # Posición de la muñeca
        
    'Cálculos intermedios'
    nx1 = np.sqrt(c[0]**2 + c[1]**2 - b**2) - a1
    # nx1 = c[0]-a1
    cz1 = c[2]-c1
    s1 = np.sqrt(nx1**2 + cz1**2)
    psi1 = np.arctan2(b, nx1 + a1) # cero

    'Calculamos las dos posibles soluciones para θ1'
    theta1_1 = np.arctan2(c[1], c[0]) - psi1                # shoulders FRONT
    theta1_2 = np.arctan2(c[1], c[0]) + 2*psi1 - np.pi      # shoulders BACK
    # .............................................................................
    # grados1_1 = -np.degrees(theta1_1)
    # grados1_2 = -np.degrees(theta1_2)

    'Cálculo de las 4 soluciones para θ2 y θ3'
    tmp_z = c[2] - c1                       # cz0 - c1
    s1_2 = nx1**2 + tmp_z**2                # s1**2
    s2_2 = (nx1 + 2 * a1)**2 + tmp_z**2     # s2**2
    kappa_2 = a2**2 + c3**2                 # k**2
    c2_2 = c2**2                            # c2**2
    'Soluciones para θ2'
    # shoulders FRONT
    theta2_1 = np.arctan2(nx1, tmp_z) - np.arccos((s1_2 + c2_2 - kappa_2) / (2 * np.sqrt(s1_2) * c2)) # elbow UP
    theta2_2 = np.arctan2(nx1, tmp_z) + np.arccos((s1_2 + c2_2 - kappa_2) / (2 * np.sqrt(s1_2) * c2)) # elbow DOWN
    # shoulders BACK
    theta2_3 = -np.arctan2(nx1 + 2 * a1, tmp_z) - np.arccos((s2_2 + c2_2 - kappa_2) / (2 * np.sqrt(s2_2) * c2)) # elbow UP
    theta2_4 = -np.arctan2(nx1 + 2 * a1, tmp_z) + np.arccos((s2_2 + c2_2 - kappa_2) / (2 * np.sqrt(s2_2) * c2)) # elbow DOWN
    # .............................................................................
    # grados2_1 = np.degrees(theta2_1)-90
    # grados2_2 = np.degrees(theta2_2)-90
    # grados2_3 = np.degrees(theta2_3)-90
    # grados2_4 = np.degrees(theta2_4)-90

    'Soluciones para θ3'
    # shoulders FRONT
    theta3_1 = np.arccos((s1_2 - c2_2 - kappa_2) / (2 * c2 * np.sqrt(kappa_2))) - np.arctan2(a2, c3)  # elbow UP
    theta3_2 = -np.arccos((s1_2 - c2_2 - kappa_2) / (2 * c2 * np.sqrt(kappa_2))) - np.arctan2(a2, c3) # elbow DOWN
    # shoulders BACK
    theta3_3 = np.arccos((s2_2 - c2_2 - kappa_2) / (2 * c2 * np.sqrt(kappa_2))) - np.arctan2(a2, c3)  # elbow UP
    theta3_4 = -np.arccos((s2_2 - c2_2 - kappa_2) / (2 * c2 * np.sqrt(kappa_2))) - np.arctan2(a2, c3) # elbow DOWN
    # .............................................................................
    # grados3_1 = -np.degrees(theta3_1)
    # grados3_2 = -np.degrees(theta3_2)
    # grados3_3 = -np.degrees(theta3_3)
    # grados3_4 = -np.degrees(theta3_4)

    'Retornar todas las combinaciones posibles de θ1, θ2 y θ3'
    solutions = [
        [theta1_1, theta2_1, theta3_1],
        [theta1_1, theta2_2, theta3_2],
        [theta1_2, theta2_3, theta3_3],
        [theta1_2, theta2_4, theta3_4]
    ]

    'Cálculo de θ4, θ5 y θ6'
    final_solutions = []
    for sol in solutions:
        theta1, theta2, theta3 = sol

        'Calcular la MATRIZ DE ROTACION DEL BRAZO (R_03)'
        c1, s1 = np.cos(theta1), np.sin(theta1)
        c2, s2 = np.cos(theta2), np.sin(theta2)
        c3, s3 = np.cos(theta3), np.sin(theta3)
        
        R_03 = np.array([
            [c1 * c2 * c3 - c1 * s2 * s3,      -s1,     c1 * c2 * s3 + c1 * s2 * c3],
            [s1 * c2 * c3 - s1 * s2 * s3,       c1,     s1 * c2 * s3 + s1 * s2 * c3],
            [         -s2 * c3 - c2 * s3,        0,              -s2 * s3 + c2 * c3]
        ])

        'Cálculo de la matriz R_36 (ORIENTACION DE LA MUÑECA)'
        R_36 = np.linalg.inv(R_03).dot(T_coordinates[:3, :3])

        'Calcular θ5'
        theta5_1 = np.arctan2(np.sqrt(1 - R_36[2, 2]**2), R_36[2, 2])
        theta5_2 = np.arctan2(-np.sqrt(1 - R_36[2, 2]**2), R_36[2, 2])

        'Calcular θ4 y θ6 para cada opción de θ5'
        ii = 0
        for theta5 in [theta5_1, theta5_2]:
            if np.abs(theta5) < 1e-6:  
                '''
                Si θ5 es cercano a cero, angulos θ4 y θ6 se simplifican:
                θ4=0 y θ6 se calcula directamente a partir de R_36
                '''
                # muneca sin invertir
                theta4 = 0
                theta6 = np.arctan2(R_36[1, 0], R_36[0, 0])
                # muneca invertida
                theta4_q = theta4 - np.pi
                theta6_q = theta6 - np.pi
            else:
                # muneca sin invertir
                theta4 = np.arctan2(R_36[1, 2], R_36[0, 2])
                theta6 = np.arctan2(R_36[2, 1], -R_36[2, 0])
                # muneca invertida
                theta4_q = theta4 - np.pi
                theta6_q = theta6 - np.pi
            #..................................................................
            ii += 1
            #..................................................................
            # Si estamos en la segunda iteración, tomar la solución invertida
            if ii == 2:
                theta4 = theta4_q
                theta6 = theta6_q
            'Convertir a grados'
            grados1 = -np.degrees(theta1)
            grados2 = np.degrees(theta2)-90
            grados3 = np.degrees(theta3)
            grados4 = -np.degrees(theta4)
            grados5 = np.degrees(theta5)
            grados6 = -np.degrees(theta6)
            #..................................................................
            'normalizar los ángulos dentro del rango [−180º,180º]'
            # if grados4 > 180:
            #     grados4 = grados4-360
            # if grados6 > 180:
            #     grados6 = grados6-360
            grados1 = (grados1 + 180) % 360 - 180
            grados2 = (grados2 + 180) % 360 - 180
            grados3 = (grados3 + 180) % 360 - 180
            grados4 = (grados4 + 180) % 360 - 180
            grados5 = (grados5 + 180) % 360 - 180
            grados6 = (grados6 + 180) % 360 - 180
            'Añadir la solución completa (θ1, θ2, θ3, θ4, θ5, θ6)'
            final_solutions.append([grados1, grados2, grados3, grados4, grados5, grados6])
    #..............................................................................
    articulaciones_inver = np.array(final_solutions)
    articulaciones_inver[np.abs(articulaciones_inver) < 1e-6] = 0 # Reemplazar los valores absolutos inferiores a 1e-6 por 0

    'Filtrar soluciones y asignar NaN donde sea necesario'
    articulaciones_filter = np.copy(articulaciones_inver)

    for i, sol in enumerate(articulaciones_inver):
        for j, angle in enumerate(sol):
            if j == 0:    # θ1
                if angle < limites['joint1'][0] or angle > limites['joint1'][1]:
                    articulaciones_filter[i, j] = np.nan
            elif j == 1:  # θ2
                if angle < limites['joint2'][0] or angle > limites['joint2'][1]:
                    articulaciones_filter[i, j] = np.nan
            elif j == 2:  # θ3
                if angle < limites['joint3'][0] or angle > limites['joint3'][1]:
                    articulaciones_filter[i, j] = np.nan
            elif j == 3:  # θ4
                if angle < limites['joint4'][0] or angle > limites['joint4'][1]:
                    articulaciones_filter[i, j] = np.nan
            elif j == 4:  # θ5
                if angle < limites['joint5'][0] or angle > limites['joint5'][1]:
                    articulaciones_filter[i, j] = np.nan
            elif j == 5:  # θ6
                if angle < limites['joint6'][0] or angle > limites['joint6'][1]:
                    articulaciones_filter[i, j] = np.nan
    
    
    
    return articulaciones_filter

def inverse_kinematic_TEKNIKER(coordinates,OPW_params,limites):
    
    'Parametros OPW del robot'
    a1 = OPW_params[0]
    a2 = OPW_params[1]
    c1 = OPW_params[2]
    c2 = OPW_params[3]
    c3 = OPW_params[4]
    c4 = OPW_params[5]
    b  = OPW_params[6]
    # .........................................................................
    T_coordinates = transformation_matrix(*coordinates)
    'Calculamos la posición del centro de la muñeca (C)'
    u_e = T_coordinates[:3, 3]  # Posición del efector final
    z_e = T_coordinates[:3, 2]  # Eje z del efector
    c = u_e - c4 * z_e  # Posición de la muñeca
        
    'Cálculos intermedios'
    nx1 = np.sqrt(c[0]**2 + c[1]**2 - b**2) - a1
    # nx1 = c[0]-a1
    cz1 = c[2]-c1
    s1 = np.sqrt(nx1**2 + cz1**2)
    psi1 = np.arctan2(b, nx1 + a1) # cero

    'Calculamos las dos posibles soluciones para θ1'
    theta1_1 = np.arctan2(c[1], c[0]) - psi1                # shoulders FRONT
    theta1_2 = np.arctan2(c[1], c[0]) + 2*psi1 - np.pi      # shoulders BACK
    # .............................................................................

    'Cálculo de las 4 soluciones para θ2 y θ3'
    tmp_z = c[2] - c1                       # cz0 - c1
    s1_2 = nx1**2 + tmp_z**2                # s1**2
    s2_2 = (nx1 + 2 * a1)**2 + tmp_z**2     # s2**2
    kappa_2 = a2**2 + c3**2                 # k**2
    c2_2 = c2**2                            # c2**2
    'Soluciones para θ2'
    # shoulders FRONT
    theta2_1 = np.arctan2(nx1, tmp_z) - np.arccos((s1_2 + c2_2 - kappa_2) / (2 * np.sqrt(s1_2) * c2)) # elbow UP
    theta2_2 = np.arctan2(nx1, tmp_z) + np.arccos((s1_2 + c2_2 - kappa_2) / (2 * np.sqrt(s1_2) * c2)) # elbow DOWN
    # shoulders BACK
    theta2_3 = np.arctan2(nx1 + 2 * a1, tmp_z) - np.arccos((s2_2 + c2_2 - kappa_2) / (2 * np.sqrt(s2_2) * c2)) # elbow UP
    theta2_4 = np.arctan2(nx1 + 2 * a1, tmp_z) + np.arccos((s2_2 + c2_2 - kappa_2) / (2 * np.sqrt(s2_2) * c2)) # elbow DOWN
    # .............................................................................

    'Soluciones para θ3'
    # shoulders FRONT
    theta3_1 = np.arccos((s1_2 - c2_2 - kappa_2) / (2 * c2 * np.sqrt(kappa_2))) - np.arctan2(a2, c3)  # elbow UP
    theta3_2 = -np.arccos((s1_2 - c2_2 - kappa_2) / (2 * c2 * np.sqrt(kappa_2))) - np.arctan2(a2, c3) # elbow DOWN
    # shoulders BACK
    theta3_3 = np.arccos((s2_2 - c2_2 - kappa_2) / (2 * c2 * np.sqrt(kappa_2))) - np.arctan2(a2, c3)  # elbow UP
    theta3_4 = -np.arccos((s2_2 - c2_2 - kappa_2) / (2 * c2 * np.sqrt(kappa_2))) - np.arctan2(a2, c3) # elbow DOWN
    # .............................................................................

    'Retornar todas las combinaciones posibles de θ1, θ2 y θ3'
    solutions = [
        [theta1_1, theta2_1, theta3_1],
        [theta1_1, theta2_2, theta3_2],
        [theta1_2, theta2_3, theta3_3],
        [theta1_2, theta2_4, theta3_4]
    ]

    'Cálculo de θ4, θ5 y θ6'
    final_solutions = []
    for sol in solutions:
        theta1, theta2, theta3 = sol

        'Calcular la MATRIZ DE ROTACION DEL BRAZO (R_03)'
        c1, s1 = np.cos(theta1), np.sin(theta1)
        c2, s2 = np.cos(theta2), np.sin(theta2)
        c3, s3 = np.cos(theta3), np.sin(theta3)
        
        R_03 = np.array([
            [c1 * c2 * c3 - c1 * s2 * s3,      -s1,     c1 * c2 * s3 + c1 * s2 * c3],
            [s1 * c2 * c3 - s1 * s2 * s3,       c1,     s1 * c2 * s3 + s1 * s2 * c3],
            [         -s2 * c3 - c2 * s3,        0,              -s2 * s3 + c2 * c3]
        ])

        'Cálculo de la matriz R_36 (ORIENTACION DE LA MUÑECA)'
        R_36 = np.linalg.inv(R_03).dot(T_coordinates[:3, :3])

        'Calcular θ5'
        theta5_1 = np.arctan2(np.sqrt(1 - R_36[2, 2]**2), R_36[2, 2])
        theta5_2 = np.arctan2(-np.sqrt(1 - R_36[2, 2]**2), R_36[2, 2])

        'Calcular θ4 y θ6 para cada opción de θ5'
        ii = 0
        for theta5 in [theta5_1, theta5_2]:
            if np.abs(theta5) < 1e-6:  
                '''
                Si θ5 es cercano a cero, angulos θ4 y θ6 se simplifican:
                θ4=0 y θ6 se calcula directamente a partir de R_36
                '''
                # muneca sin invertir
                theta4 = 0
                theta6 = np.arctan2(R_36[1, 0], R_36[0, 0])
                # muneca invertida
                theta4_q = theta4 - np.pi
                theta6_q = theta6 - np.pi
            else:
                # muneca sin invertir
                theta4 = np.arctan2(R_36[1, 2], R_36[0, 2])
                theta6 = np.arctan2(R_36[2, 1], -R_36[2, 0])
                # muneca invertida
                theta4_q = theta4 - np.pi
                theta6_q = theta6 - np.pi
            #......................................................................
            ii += 1
            #......................................................................
            # Si estamos en la segunda iteración, tomar la solución invertida
            if ii == 2:
                theta4 = theta4_q
                theta6 = theta6_q
            'Convertir a grados'
            grados1 = -np.degrees(theta1)-0.044
            grados2 = np.degrees(theta2) -90.028
            grados3 = np.degrees(theta3) + 0.011
            grados4 = -np.degrees(theta4)- 0.006
            grados5 = np.degrees(theta5)
            grados6 = -np.degrees(theta6)
            #......................................................................
            'Añadir la solución completa (θ1, θ2, θ3, θ4, θ5, θ6)'
            final_solutions.append([grados1, grados2, grados3, grados4, grados5, grados6])
    #..............................................................................
    articulaciones_inver = np.array(final_solutions)
    articulaciones_inver[np.abs(articulaciones_inver) < 1e-6] = 0 # Reemplazar los valores absolutos inferiores a 1e-6 por 0

    'Filtrar soluciones y asignar NaN donde sea necesario'
    articulaciones_filter = np.copy(articulaciones_inver)

    for i, sol in enumerate(articulaciones_inver):
        for j, angle in enumerate(sol):
            if j == 0:    # θ1
                if angle < limites['joint1'][0] or angle > limites['joint1'][1]:
                    articulaciones_filter[i, j] = np.nan
            elif j == 1:  # θ2
                if angle < limites['joint2'][0] or angle > limites['joint2'][1]:
                    articulaciones_filter[i, j] = np.nan
            elif j == 2:  # θ3
                if angle < limites['joint3'][0] or angle > limites['joint3'][1]:
                    articulaciones_filter[i, j] = np.nan
            elif j == 3:  # θ4
                if angle < limites['joint4'][0] or angle > limites['joint4'][1]:
                    articulaciones_filter[i, j] = np.nan
            elif j == 4:  # θ5
                if angle < limites['joint5'][0] or angle > limites['joint5'][1]:
                    articulaciones_filter[i, j] = np.nan
            elif j == 5:  # θ6
                if angle < limites['joint6'][0] or angle > limites['joint6'][1]:
                    articulaciones_filter[i, j] = np.nan
    
    
    
    return articulaciones_filter

def concretar_solucion(opciones, invertir_base=False, invertir_codo=False, invertir_muneca=False):
    """
    Función para seleccionar la solución de las articulaciones en base a las opciones de invertir base, codo y muñeca.
    
    Parámetros:
    - opciones: un array numpy de todas las soluciones posibles (6 articulaciones por fila)
    - invertir_base: bool, si es True, se seleccionan las soluciones con base invertida
    - invertir_codo: bool, si es True, se seleccionan las soluciones con codo invertido
    - invertir_muneca: bool, si es True, se seleccionan las soluciones con muñeca invertida
    
    Retorno:
    - ARTICULACIONES_INVERSE: una lista con la configuración de las articulaciones seleccionada
    """
    
    # Selección de las soluciones posibles en base a invertir_base
    if invertir_base:
        posibles_sol = opciones[4:8, :]
    else:
        posibles_sol = opciones[0:4, :]
    
    # Selección de las soluciones posibles en base a invertir_codo
    if invertir_codo:
        posibles_sol = posibles_sol[2:4, :]
    else:
        posibles_sol = posibles_sol[0:2, :]
    
    # Selección de las soluciones posibles en base a invertir_muneca
    if invertir_muneca:
        posibles_sol = posibles_sol[1:2, :]
    else:
        posibles_sol = posibles_sol[0:1, :]
    
    # Asignar las articulaciones seleccionadas
    J1 = posibles_sol[0, 0]
    J2 = posibles_sol[0, 1]
    J3 = posibles_sol[0, 2]
    J4 = posibles_sol[0, 3]
    J5 = posibles_sol[0, 4]
    J6 = posibles_sol[0, 5]
    
    # Resultado final
    solucion_articulaciones = np.array([J1, J2, J3, J4, J5, J6])
    
    return solucion_articulaciones

'''
*******************************************************************************
                       PROGRAMA PRINCIPAL
*******************************************************************************
'''

'Ejemplo: configuracion del robot'
articulaciones_inicio = [
    126.710,                # J1 (grados)
    -98.200,                # J2 (grados)
    117.320,                # J3 (grados)
    0.060,                  # J4 (grados)
    69.910,                 # J5 (grados)
    -82.650                 # J6 (grados)
    ]

print('\nInput: [A1 {:.3f}, A2 {:.3f}, A3 {:.3f}, A4 {:.3f}, A5 {:.3f}, A6 {:.3f}]'.format(*articulaciones_inicio))
print('BASE: [X {:.3f}, Y {:.3f}, Z {:.3f}, A {:.3f}, B {:.3f}, C {:.3f}]'.format(*BASE))
print('TOOL: [X {:.3f}, Y {:.3f}, Z {:.3f}, A {:.3f}, B {:.3f}, C {:.3f}]'.format(*TOOL))
print('\n------------------------------------------------------------')
print('J1 invertido: '+str(invertir_base))
print('J3 invertido: '+str(invertir_codo))
print('J4 invertido: '+str(invertir_muneca))
print('------------------------------------------------------------')
'''
*******************************************************************************
                        DIRECT KINEMATIC
*******************************************************************************
'''
nominal = False
dh_params = dh_nominales
# dh_params = dh_estimados # TEKNIKER LASER TRACKER

# .............................................................................
'Cinematica directa' # pose robot BASE[0] y TOOL[0]
if dh_params.shape == dh_nominales.shape:
    if (dh_params == dh_nominales).all():
        print('DH NOMINALES')
        nominal = True
        T_robot = direct_kinematic(articulaciones_inicio,dh_params)
    else:
        print('DH CALCULADOS')
        T_robot = direct_kinematic_TEKNIKER(articulaciones_inicio,dh_params)
else:
    print('DH CALCULADOS')
    T_robot = direct_kinematic_TEKNIKER(articulaciones_inicio,dh_params) 
# .............................................................................
'Tranformacion coordenadas'
BASE_transformation = transformation_matrix(*BASE)
BASE_transformation_inv = np.linalg.inv(BASE_transformation)
TOOL_transformation = transformation_matrix(*TOOL)
TOOL_transformation_inv = np.linalg.inv(TOOL_transformation)
# .............................................................................
T_final = BASE_transformation_inv @ T_robot @ TOOL_transformation # pose robot con BASE'y TOOL'
# .............................................................................
'extraer valores coordenadas'
X_tool = T_final[0,3]
Y_tool = T_final[1,3]
Z_tool = T_final[2,3]
A_tool, B_tool, C_tool = extract_euler_angles(T_final)

print('\nDIRECT kinematic: [X {:.3f}, Y {:.3f}, Z {:.3f}, A {:.3f}, B {:.3f}, C {:.3f}]'.format(X_tool,Y_tool,Z_tool,A_tool,B_tool,C_tool))

''''
*******************************************************************************
                            INVERSE KINEMATIC
*******************************************************************************
'''

'Obtener la pose del extremo del robot en la base del robot'
T_robot_calc = BASE_transformation @T_final @ TOOL_transformation_inv  # pose robot BASE[0] Y TOOL[0]
# .............................................................................
# (obtener XYZABC a partir de la matriz transformacion)
'extraer valores coordenadas'
X_input = T_robot_calc[0,3]
Y_input = T_robot_calc[1,3]
Z_input = T_robot_calc[2,3]
A_input, B_input, C_input = extract_euler_angles(T_robot_calc)
# print('\ninput: [X {:.3f}, Y {:.3f}, Z {:.3f}, A {:.3f}, B {:.3f}, C {:.3f}]'.format(X_input,Y_input,Z_input,A_input,B_input,C_input))
# .............................................................................
coordinates = np.array([X_input,Y_input,Z_input,A_input,B_input,C_input])
'POSIBLES SOLUCIONES - CINEMATICA INVERSA'
OPW_parameters = OPW_nominales
soluciones = inverse_kinematic(coordinates,OPW_parameters,limits)

'concretar solucion'
articulaciones_inverse = concretar_solucion(soluciones, invertir_base, invertir_codo, invertir_muneca)
print('\nINVERSE kinematics: [A1 {:.3f}, A2 {:.3f}, A3 {:.3f}, A4 {:.3f}, A5 {:.3f}, A6 {:.3f}]'.format(*articulaciones_inverse))



# .............................................................................
# 'Calculamos la posición del centro de la muñeca (C)'
# u_e = T_robot_calc[:3, 3]  # Posición del efector final
# z_e = T_robot_calc[:3, 2]  # Eje z del efector
# c = u_e - c4 * z_e  # Posición de la muñeca
    
# 'Cálculos intermedios'
# nx1 = np.sqrt(c[0]**2 + c[1]**2 - b**2) - a1
# # nx1 = c[0]-a1
# cz1 = c[2]-c1
# s1 = np.sqrt(nx1**2 + cz1**2)
# psi1 = np.arctan2(b, nx1 + a1) # cero

# 'Calculamos las dos posibles soluciones para θ1'
# theta1_1 = np.arctan2(c[1], c[0]) - psi1                # shoulders FRONT
# theta1_2 = np.arctan2(c[1], c[0]) + 2*psi1 - np.pi      # shoulders BACK
# # .............................................................................
# # grados1_1 = -np.degrees(theta1_1)
# # grados1_2 = -np.degrees(theta1_2)

# 'Cálculo de las 4 soluciones para θ2 y θ3'
# tmp_z = c[2] - c1                       # cz0 - c1
# s1_2 = nx1**2 + tmp_z**2                # s1**2
# s2_2 = (nx1 + 2 * a1)**2 + tmp_z**2     # s2**2
# kappa_2 = a2**2 + c3**2                 # k**2
# c2_2 = c2**2                            # c2**2
# 'Soluciones para θ2'
# # shoulders FRONT
# theta2_1 = np.arctan2(nx1, tmp_z) - np.arccos((s1_2 + c2_2 - kappa_2) / (2 * np.sqrt(s1_2) * c2)) # elbow UP
# theta2_2 = np.arctan2(nx1, tmp_z) + np.arccos((s1_2 + c2_2 - kappa_2) / (2 * np.sqrt(s1_2) * c2)) # elbow DOWN
# # shoulders BACK
# theta2_3 = np.arctan2(nx1 + 2 * a1, tmp_z) - np.arccos((s2_2 + c2_2 - kappa_2) / (2 * np.sqrt(s2_2) * c2)) # elbow UP
# theta2_4 = np.arctan2(nx1 + 2 * a1, tmp_z) + np.arccos((s2_2 + c2_2 - kappa_2) / (2 * np.sqrt(s2_2) * c2)) # elbow DOWN
# # .............................................................................
# # grados2_1 = np.degrees(theta2_1)-90
# # grados2_2 = np.degrees(theta2_2)-90
# # grados2_3 = np.degrees(theta2_3)-90
# # grados2_4 = np.degrees(theta2_4)-90

# 'Soluciones para θ3'
# # shoulders FRONT
# theta3_1 = np.arccos((s1_2 - c2_2 - kappa_2) / (2 * c2 * np.sqrt(kappa_2))) - np.arctan2(a2, c3)  # elbow UP
# theta3_2 = -np.arccos((s1_2 - c2_2 - kappa_2) / (2 * c2 * np.sqrt(kappa_2))) - np.arctan2(a2, c3) # elbow DOWN
# # shoulders BACK
# theta3_3 = np.arccos((s2_2 - c2_2 - kappa_2) / (2 * c2 * np.sqrt(kappa_2))) - np.arctan2(a2, c3)  # elbow UP
# theta3_4 = -np.arccos((s2_2 - c2_2 - kappa_2) / (2 * c2 * np.sqrt(kappa_2))) - np.arctan2(a2, c3) # elbow DOWN
# # .............................................................................
# # grados3_1 = -np.degrees(theta3_1)
# # grados3_2 = -np.degrees(theta3_2)
# # grados3_3 = -np.degrees(theta3_3)
# # grados3_4 = -np.degrees(theta3_4)

# 'Retornar todas las combinaciones posibles de θ1, θ2 y θ3'
# solutions = [
#     [theta1_1, theta2_1, theta3_1],
#     [theta1_1, theta2_2, theta3_2],
#     [theta1_2, theta2_3, theta3_3],
#     [theta1_2, theta2_4, theta3_4]
# ]

# # articulaciones = [
# #     [grados1_1, grados2_1, grados3_1],
# #     [grados1_1, grados2_2, grados3_2],
# #     [grados1_2, grados2_3, grados3_3],
# #     [grados1_2, grados2_4, grados3_4]
# # ]

# 'Cálculo de θ4, θ5 y θ6'
# final_solutions = []
# for sol in solutions:
#     theta1, theta2, theta3 = sol

#     'Calcular la MATRIZ DE ROTACION DEL BRAZO (R_03)'
#     c1, s1 = np.cos(theta1), np.sin(theta1)
#     c2, s2 = np.cos(theta2), np.sin(theta2)
#     c3, s3 = np.cos(theta3), np.sin(theta3)
    
#     R_03 = np.array([
#         [c1 * c2 * c3 - c1 * s2 * s3,      -s1,     c1 * c2 * s3 + c1 * s2 * c3],
#         [s1 * c2 * c3 - s1 * s2 * s3,       c1,     s1 * c2 * s3 + s1 * s2 * c3],
#         [         -s2 * c3 - c2 * s3,        0,              -s2 * s3 + c2 * c3]
#     ])

#     'Cálculo de la matriz R_36 (ORIENTACION DE LA MUÑECA)'
#     R_36 = np.linalg.inv(R_03).dot(T_robot_calc[:3, :3])

#     'Calcular θ5'
#     theta5_1 = np.arctan2(np.sqrt(1 - R_36[2, 2]**2), R_36[2, 2])
#     theta5_2 = np.arctan2(-np.sqrt(1 - R_36[2, 2]**2), R_36[2, 2])

#     'Calcular θ4 y θ6 para cada opción de θ5'
#     ii = 0
#     for theta5 in [theta5_1, theta5_2]:
#         if np.abs(theta5) < 1e-6:  
#             '''
#             Si θ5 es cercano a cero, angulos θ4 y θ6 se simplifican:
#             θ4=0 y θ6 se calcula directamente a partir de R_36
#             '''
#             # muneca sin invertir
#             theta4 = 0
#             theta6 = np.arctan2(R_36[1, 0], R_36[0, 0])
#             # muneca invertida
#             theta4_q = theta4 - np.pi
#             theta6_q = theta6 - np.pi
#         else:
#             # muneca sin invertir
#             theta4 = np.arctan2(R_36[1, 2], R_36[0, 2])
#             theta6 = np.arctan2(R_36[2, 1], -R_36[2, 0])
#             # muneca invertida
#             theta4_q = theta4 - np.pi
#             theta6_q = theta6 - np.pi
#         #......................................................................
#         ii += 1
#         #......................................................................
#         # Si estamos en la segunda iteración, tomar la solución invertida
#         if ii == 2:
#             theta4 = theta4_q
#             theta6 = theta6_q
#         'Convertir a grados'
#         grados1 = -np.degrees(theta1)
#         grados2 = np.degrees(theta2)-90
#         grados3 = np.degrees(theta3)
#         grados4 = -np.degrees(theta4)
#         grados5 = np.degrees(theta5)
#         grados6 = -np.degrees(theta6)
#         #......................................................................
#         'Añadir la solución completa (θ1, θ2, θ3, θ4, θ5, θ6)'
#         final_solutions.append([grados1, grados2, grados3, grados4, grados5, grados6])
# #..............................................................................
# articulaciones_inver = np.array(final_solutions)
# articulaciones_inver[np.abs(articulaciones_inver) < 1e-6] = 0 # Reemplazar los valores absolutos inferiores a 1e-6 por 0

# 'Filtrar soluciones y asignar NaN donde sea necesario'
# articulaciones_filter = np.copy(articulaciones_inver)

# for i, sol in enumerate(articulaciones_inver):
#     for j, angle in enumerate(sol):
#         if j == 0:    # θ1
#             if angle < limits['joint1'][0] or angle > limits['joint1'][1]:
#                 articulaciones_filter[i, j] = np.nan
#         elif j == 1:  # θ2
#             if angle < limits['joint2'][0] or angle > limits['joint2'][1]:
#                 articulaciones_filter[i, j] = np.nan
#         elif j == 2:  # θ3
#             if angle < limits['joint3'][0] or angle > limits['joint3'][1]:
#                 articulaciones_filter[i, j] = np.nan
#         elif j == 3:  # θ4
#             if angle < limits['joint4'][0] or angle > limits['joint4'][1]:
#                 articulaciones_filter[i, j] = np.nan
#         elif j == 4:  # θ5
#             if angle < limits['joint5'][0] or angle > limits['joint5'][1]:
#                 articulaciones_filter[i, j] = np.nan
#         elif j == 5:  # θ6
#             if angle < limits['joint6'][0] or angle > limits['joint6'][1]:
#                 articulaciones_filter[i, j] = np.nan

# '''
#     CONCRETAR SOLUCION
# '''

# todas_opciones = soluciones
# # todas_opciones = articulaciones_filter # limites de ejes

# if invertir_base:
#     posibles_sol = todas_opciones[4:8,:]
# else:
#     posibles_sol = todas_opciones[0:4,:]
    

# if invertir_codo:
#     posibles_sol = posibles_sol[2:4,:]
# else:
#     posibles_sol = posibles_sol[0:2,:]
    
# if invertir_muneca:
#     posibles_sol = posibles_sol[1:2,:]
# else:
#     posibles_sol = posibles_sol[0:1,:]

# J1 = posibles_sol[0,0]
# J2 = posibles_sol[0,1]
# J3 = posibles_sol[0,2]
# J4 = posibles_sol[0,3]
# J5 = posibles_sol[0,4]
# J6 = posibles_sol[0,5]
# ARTICULACIONES_INVERSE = [J1, J2, J3, J4, J5, J6]