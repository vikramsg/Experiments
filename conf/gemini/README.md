# Gemini

## Install

```sh
npm install -g @google/gemini-cli
```

## Vertex AI

Setup the following in .env to use Vertex AI. 

```
# Don't set it to 1. Won't work
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT
# Don't set it to global. Won't work
GOOGLE_CLOUD_LOCATION=us-central1
```

Authenticate using

```
gcloud auth login
# Will open a browser window. Hopefully we need to do this only once.
gcloud auth application-default login
gcloud config set project $GOOGLE_CLOUD_PROJECT
```
