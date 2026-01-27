# Aprendizaje de Máquina 2026

Este repositorio contiene ejercicios y proyectos para el curso de Aprendizaje de Máquina. El paquete incluye las bibliotecas y herramientas comunes de ML para análisis de datos, visualización y desarrollo de modelos de aprendizaje de máquina.

## Prerequisitos

Antes de comenzar, asegúrate de tener instalado lo siguiente en tu sistema:

### Software Requerido

1. **Git**
   - Verifica tu versión de Git: `git --version`
   - Descarga desde: [git-scm.com](https://git-scm.com/downloads)

2. **Editor de Código o IDE**
   - [Visual Studio Code](https://code.visualstudio.com/)

### Cuenta de GitHub

- Necesitarás una cuenta de GitHub para hacer fork del repositorio
- Crea una cuenta gratuita en: [github.com](https://github.com/join)

## Cómo usar este repositorio?
1. Haz un fork de este repositorio. Define el nombre como ML26_[nombre_de_equipo]
![create new fork](./imgs/forking.png)
3. Ve a tu nuevo repositorio y copia el link en code->Local-> HTTPS
![create new fork](./imgs/clone.png)
4. Clona tu forked repo localmente
```
git clone [http link]
```
5. Agrega el repositorio original como upstream

```
git remote set-url upstream https://github.com/JessicaBorja/ml26_01.git
```

## Instalación

Este proyecto utiliza [uv](https://github.com/astral-sh/uv) de Astral para una gestión de paquetes de Python rápida y confiable.

### 1. Instalar uv

#### Windows

```powershell
# Usando PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

O usando pip:
```powershell
pip install uv
```

#### macOS y Linux

```bash
# Usando curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

O usando pip:
```bash
pip install uv
```

### 2. Instalar el Proyecto

Una vez que uv esté instalado, navega al directorio del proyecto y ejecuta:

```bash
uv sync
```

Este comando:
- Crea automáticamente un entorno virtual en `.venv` si no existe
- Instala todas las dependencias especificadas en `pyproject.toml`
- Crea un archivo `uv.lock` para garantizar instalaciones reproducibles
- Instala el paquete en modo editable

### 3. Activar el Entorno Virtual

Después de `uv sync`, activa el entorno virtual:

```bash
# En Windows:
.venv\Scripts\activate

# En macOS/Linux:
source .venv/bin/activate
```

## Actualizar tu repositorio
Para actualizar tu repositorio local con los cambios del remoto ejecuta los siguientes comandos.

### Sincronizar repo local con remoto
### (Local -> Remoto)
Para mandar a github los commits que tengas localmente que no estén en el repositorio remoto, escribe en tu consola de comandos dentro de tu repositorio local:

```
git push
```

### (Remoto -> local)
Para traer a tu repositorio local los commits que se encuentran en el repositorio remoto, escribe en tu consola de comandos dentro de tu repositorio local:

```
git pull
```

### Sincronizar repo local y remoto con el upstream (original)
Los siguientes comandos:

1. (fetch) Buscan los cambios en el upstream
2. (merge) Integran los cambios del upstream a tu repositorio local
3. (push) Sincronizan los cambios del repositorio local (que vienen del upstream) a tu repositorio remoto en github

```
git fetch upstream
git merge upstream/master
git push
```
