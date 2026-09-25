# Bilingual SMS Spam Detector

This project detects spam or ham SMS messages in English, Hindi, and Hinglish.

## Start the website

### Option A: two PowerShell terminals (recommended)

**Terminal 1: start the backend**

```powershell
Set-Location "C:\spam sms  detector"
python app.py
```

Leave this terminal running.

**Terminal 2: start the frontend**

```powershell
Set-Location "C:\spam sms  detector\frontend"
npm install
npm run dev
```

Run `npm install` only once. Leave this terminal running too.

Open the website at:

```text
http://127.0.0.1:5173
```

### Option B: one PowerShell terminal

Run this entire block from one PowerShell terminal:

```powershell
Set-Location "C:\spam sms  detector"
$backend = Start-Job -ScriptBlock { Set-Location "C:\spam sms  detector"; python app.py }
Set-Location ".\frontend"
npm install
npm run dev
```

The backend runs in the background and Vite runs in the foreground. Open
`http://127.0.0.1:5173` when Vite shows its local URL.

To stop the one-terminal setup, press `Ctrl+C`, then run:

```powershell
Get-Job | Stop-Job
Get-Job | Remove-Job
```

## Check the backend

Open this address in a browser:

```text
http://127.0.0.1:5000/health
```

It should show `"status": "ok"`, `"model_loaded": true`, and
`"vectorizer_loaded": true`.

## Important

Never run `npm run dev` from `C:\spam sms  detector`. That folder has no
`package.json`. Run it from `C:\spam sms  detector\frontend`, or use Option B.