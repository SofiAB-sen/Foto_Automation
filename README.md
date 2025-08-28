# Character Detection Automation

Este proyecto utiliza modelos YOLOv11 con Ultralytics para realizar reconocimiento automático de numeros de placas vehiculares (ANPR - Automatic Number Plate Recognition) a partir de imágenes. 

---

## ⚙️ Requisitos

### 🔹 Requisitos del sistema

- Python 3.10
- Conda o Miniconda

### 🔹 Crear entorno Conda usando requirements.txt

```bash
conda create -n yolo-detect python=3.10
conda activate yolo-detect
pip install -r [requirements.txt](http://_vscodecontentref_/1)
```
Sin requirements.txt:

```bash
conda create -n yolo-detect python=3.10
conda activate yolo-detect

# Dependencias básicas
conda install -c conda-forge numpy<2 pillow matplotlib tqdm pyyaml pandas seaborn pip requests opencv=4.8.0.76 

#Ultralytics y dependencias adicionales
pip install requests psutil py-cpuinfo "ultralytics-thop>=2.0.0"
```

🚀 Cómo usar

Ejecuta el script Detect.py pasando los argumentos: --process_id --image_path --detected_plate --endpoint_url

```bash
python3 Detect.py --process_id 1 --image_path "https://www.pruebaderuta.com/wp-content/uploads/2016/03/placa-amarilla.jpg" --detected_plate CVY000 --endpoint_url "https://webhook.site/c016b479-3691-436a-bd36-bcd1c6ead397"
```
