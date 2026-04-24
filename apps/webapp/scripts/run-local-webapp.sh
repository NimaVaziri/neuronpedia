#!/bin/sh
set -eu

cd /root/ce-qweb3-4b/apps/webapp
echo "Starting Neuronpedia webapp at $(date -Is)" > /tmp/neuronpedia-webapp.log
exec npm run dev:localhost >> /tmp/neuronpedia-webapp.log 2>&1
