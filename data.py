import pandas as pd
import tensorflow as tf

products_tabla = pd.DataFrame.from_records([
    ['Mandarin', 0, 4, 40, 1, 0, 10, 1, 'u'],
    ['Queso', 0, 8, 350, 28, 26, 2, 100, 'gr'],
    ['Huevo duro', 0, 8, 155, 13, 11, 1, 1, 'u'],
    ['Pizza de queso personal', 0, 3, 903, 36, 47, 81, 1, 'u'],
    ['Chocolatada 250 ml', 0, 3, 160, 7, 6,20,1,'u'],
    ['Banana ', 0, 4, 89, 1, 0, 23,1, 'u'],
])
products_tabla.columns = ['Nombre', 'Min', 'Max', 'Calorias', 'Gr_Prot', 'Gr_Grasa', 'Gr_Carb','Cantidad','Unidad']

prot_data=tf.constant(products_tabla['Gr_Prot'],dtype='float32')
fat_data = tf.constant(products_tabla['Gr_Grasa'],dtype='float32')
carb_data = tf.constant(products_tabla['Gr_Carb'],dtype='float32')
min_data = tf.constant(products_tabla['Min'])
max_data = tf.constant(products_tabla['Max'])
quantity_data = tf.constant(products_tabla['Cantidad'],dtype='float32')

prot_cal_p_gram = tf.constant(4,dtype='float32')
carb_cal_p_gram = tf.constant(4,dtype='float32')
fat_cal_p_gram = tf.constant(9,dtype='float32')

Genome = tf.constant(value=tf.range(len(products_tabla.index)))