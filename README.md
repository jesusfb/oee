# OEE & Downtime Dashboard

Dashboard **Streamlit** de OEE, Six Big Losses, paros (Pareto, MTBF/MTTR), turnos, calidad y what-if. Datos de muestra, simulación o CSV propios (esquema MES en la pestaña *Data & schema*).

## Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.10+ (local). En **Streamlit Cloud** usa 3.11 o superior.

## Desplegar en Streamlit Community Cloud (GitHub)

1. Sube este proyecto a un repositorio en **GitHub** (público o privado; la cuenta gratuita de Streamlit admite ambos).
2. Entra en [share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub.
3. **Create app** → elige el repo, rama `main`, **Main file path:** `app.py`.
4. En **Advanced settings** → Python **3.11** (o 3.12) si el primer deploy falla por versión.
5. **Deploy**. El build instala `requirements.txt` de la raíz.

No hace falta `secrets.toml` para la planta de muestra. Si más adelante conectas SQL o API keys, usa *Settings → Secrets* en la app.

### Subir a GitHub (primera vez)

```powershell
cd C:\Users\jesus\Colef2025\jfb\OEE\oee-downtime-dashboard
git init -b main
git add -A
git commit -m "OEE downtime dashboard for Streamlit Cloud"
```

Crea un repo vacío en GitHub (sin README), luego:

```powershell
git remote add origin https://github.com/TU_USUARIO/oee-downtime-dashboard.git
git push -u origin main
```

Vuelve a share.streamlit.io y conecta ese repo.

## Estructura

| Ruta | Uso |
|------|-----|
| `app.py` | UI Streamlit (entrypoint en Cloud) |
| `oee/` | Métricas, gráficos, E/S CSV |
| `data/sample/` | Datos de ejemplo |
| `.streamlit/config.toml` | Tema oscuro |