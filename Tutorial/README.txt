Captura 1 - Conectandose al repositorio

Primero hay que descargar el cliente de escritorio para GitHub desde la página https://desktop.github.com/

Luego lo instalan

Cuando esté les va a pedir el usuario y contraseña

Cuando accedan busquen el repositorio con esta URL: https://github.com/rdgagliano/DataScience-Grupo3.git

Cuando les aparezca tienen que clonarlo a su pc local

En la captura 1:
1- Es la URL del repositorio
2- Es la dirección en su file system local en donde va a clonar el repositorio, pueden cambiarlo y ponerlo donde les parezca mas comodo con el botón Choose

Captura 2 - Pantalla Principal

Cuando están conectados al repo, le aparece esa pantalla donde se trabaja

En la captura 2:
1- Es el listado de cambios que el programa detecta entre su copia local contra la del servidor. En la foto, el servidor no tiene nada y yo agregué el archivo README.txt. Al lado del nombre del archivo hay un signo mas en verde, quiere decir que el archivo fue agregado al repositorio.

2- El botón commit sube sus cambios al repositorio, en este caso lo que haría es enviar mi archivo README.txt al servidor.

3- Ahí se pone la descripción del commit. Lo mejor es siempre poner un descripción de lo que se está commiteando, así todos tenemos idea de qué es lo que se aplicó con ese cambio. Una oración breve, que describa bien el commit basta. La foto no la tiene, pero pude haber puesto algo como "Subo un archivo README explicando como usar GitHub". En el campo de texto de arriba de la descripción tienen el título para el commit, ahí me lo autocompletó, pero podemos poner cualquier cosa. También conviene algo que sea descriptivo del cambio que subimos.

4- En esa zona se muestra las diferencias entre el servidor y la versión local para el archivo seleccionado. Ahí no me muestra nada porque el archivo README está vacío

5- Es el nombre del repositorio seleccionado, al cuál van a ir los commits. No creo q esto nos sirva, porque calculo que vamos a usar solo un repositorio. Pero si manejamos varios, ahí podemos elegir entre los distintos repositorios en los que trabajemos.

6- Es el nombre del branch actual, al cuál van a ir los commits. Por defecto el repositorio viene con el branch 'master'. Un branch sirve para manejar diferentes versiones de entrega del código. No sé si esto lo usaremos, pero por ej. cuando terminemos la entrega 1. Podemos crear un branch para esa entrega, entonces pasaríamos a tener 2 branches activos: master y entrega1 (por ej). Esto nos permite tener dos versiones del código: una para la entrega 1 y otra para seguir trabajando sin problemas. Esto nos permite seguir trabajando tranquilamente para la versión 1 sin tener posibles cambios agregados para la entrega 2 (que ya que lo tenemos separados, los haríamos en master). Cuando se hacen cambios en el branch entrega1 y quedan aprobados, pueden pasarse al branch master, así quedan aplicados ahí también y se puede seguir trabajando con el código actualizado en master.