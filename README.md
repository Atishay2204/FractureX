# FractureX

FractureX is an educational X-ray fracture screening demo. The React/Vite app
uses a FastAPI service to run the YOLO model. Capacitor packages the React app
for Android and iOS and connects its camera to the existing `/predict` endpoint.

> This is not a medical diagnosis tool. X-rays must be reviewed by a qualified clinician.

## Run the API

From the repository root, with the model checkpoint available:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

The active checkpoint is `best_binary_fracture.pt`. Set `FRACTURE_MODEL_PATH`
to use another compatible model.

## Run the React app

In a second terminal:

```powershell
cd frontend
Copy-Item .env.example .env
npm install
npm run dev -- --host 0.0.0.0
```

Set `VITE_API_URL` in `frontend/.env` to the API address. Use
`http://localhost:8000` in a desktop browser, `http://10.0.2.2:8000` in the
Android emulator, or `http://<computer-LAN-IP>:8000` on a physical phone on
the same Wi-Fi network.

## Build the Android app

Install Android Studio and its Android SDK, then run these commands from
`frontend/`:

```powershell
npm run build
npx cap add android
npx cap sync android
npx cap open android
```

`npx cap add android` is only needed once. Use Android Studio to run on a
device or create a signed release. For iOS, use a Mac with Xcode and run:

```sh
npx cap add ios
npm run mobile:sync
npx cap open ios
```

Set the camera and photo library usage descriptions in the iOS app's
`Info.plist` before building.

For a published build, configure `VITE_API_URL` with the HTTPS address of a
deployed FastAPI service that has the model checkpoint. The React app and API
can be deployed separately.

Images are processed by the API and are not intentionally written to disk by
this demo. Do not use it to make treatment decisions.

## Project layout

- `api.py`: FastAPI inference endpoint.
- `frontend/`: React/Vite web app and Capacitor Android project configuration.
- `best_binary_fracture.pt`: active binary fracture checkpoint.
- `best_verified_light.pt`: legacy six-class checkpoint.
- `src/`, `notebooks/`, `outputs/`: model training and evaluation materials.
