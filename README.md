# mvp-mlpractico
MVP del curso Machine Learning Práctico de Pablo Zivic. El plan es construirlo de forma colaborativa.
Partimos del código que fue desarrollando y explicando en las clases
Tendremos por un lado los archivos limpios del curso
Por el otro los desarrollos de: 
  /arquitectura 
  /modelado
  /presentación

# Como empezar

### Arquitectura

Dentro de la carpeta [Arquitectura](mlp_mvp/arquitectura) podemos encontrar la ssiguientes carpetas:

 - **data:** Donde depositamos los files de input.
 - **model:** Carpeta donde podremos guardar los modelos serializados (pkl).
 - **prediction_output:** Files de salida con predicciones.
 - **training_scripts:** scripts de entrenamiento.
 - **lib:** Carpeta con los transformers y scripts de armados de los dataframes.
 - **notebooks:** Notebooks de prueba para probar scripts, bajar files o hacer predicciones. Basado en los notebooks del curso.


### Instalación del entorno
Dentro de la carpeta Arquiectura encontramos un file de **requirements.txt** con las versiones necesarias para correr los primeros modelos.

Ejecutar en una consola (En linux)

Ver si tienen estas librerias
```
sudo apt install -y apparmor apturl 
sudo apt-get install swig
```

Luego creamos el enviroment
```
virtualenv ml_env
source ml_env/bin/activate
```
Instalamos las librerias correspondientes (de usar una nueva, actualizar la lista).
```
pip install -r requirements.txt --use-feature=2020-resolver
```

Si así lo desean pueden copiar este procedimiento para el resto de las etapas, en cada notebook (por el momento 2), tenemos para usar las **libs** de feature engineering importándolas con el código adaptado para el proyecto o simplemente desde cualquier script de python copiando la misma lógica de los notebooks. 


### Instalación del entorno en raíz (fuera de arquitectura)
Desde el raíz de proyecto tenemos otra forma de trabajar

```
python setup.py develop
```

Luego desde cualquier jupyter notebook van a poder hacer esto

```
from mlp_mvp.transformers import CrewFeatures

```


# Cómo contribuir?

* Hacé un checkout del branch develop: `git checkout develop`
* Asegurate de estar al día: `git pull origin HEAD`
* Crea un branch nuevo con un nombre descriptivo. Si vas a hacer el modelo de rating podrías poner: `git checkout feature/rating_prediction`
* Hacé tu trabajo pusheando en ese branch: `git push origin HEAD`
* Crea un [pull request](https://github.com/pabloromero17143/mvp-mlpractico/pulls) de tu branch al branch `develop`
* Para ver un pull requests de ejemplo mira [acá](https://github.com/pabloromero17143/mvp-mlpractico/pull/1)

*
