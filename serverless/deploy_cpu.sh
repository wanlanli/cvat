#!/bin/bash
# Sample commands to deploy nuclio functions on CPU

set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FUNCTIONS_DIR=${1:-$SCRIPT_DIR}

EXPORT_DIR="/home/liwa/data/nuclio_images"
mkdir -p "$EXPORT_DIR"

export DOCKER_BUILDKIT=1

# docker build -t cvat.openvino.base "$SCRIPT_DIR/openvino/base"
# docker build -f "$SCRIPT_DIR/Dockerfile.dlbase" -t cvat-dl-base:latest .

# docker save cvat.openvino.base -o "$EXPORT_DIR/cvat.openvino.base.tar"
# docker rmi cvat.openvino.base
# docker load -i "$EXPORT_DIR/cvat.openvino.base.tar"
echo "aaaaa"

/home/liwa/nuctl create project cvat --platform local

shopt -s globstar

for func_config in "$FUNCTIONS_DIR"/**/function.yaml
do
    echo "bbbbb"
    func_root="$(dirname "$func_config")"
    func_rel_path="$(realpath --relative-to="$SCRIPT_DIR" "$(dirname "$func_root")")"
    # image_name="cvat.${func_rel_path//\//.}.base"

    if [ -f "$func_root/Dockerfile" ]; then
    # if [ ! -f "$EXPORT_DIR/$image_name.tar" ]; then
        echo "cccccc,,,,$func_root/Dockerfile"
        # echo "cvat.${func_rel_path//\//.}.base"
        docker build -t "cvat.${func_rel_path//\//.}.base" "$func_root"

        # # docker build -t "$image_name" "$func_root"

        # # Save the image to EXPORT_DIR
        # docker save "$image_name" -o "$EXPORT_DIR/$image_name.tar"

        # # Remove local image to free space
        # docker rmi "$image_name"

    fi
    # Load the image from disk before deploy
    # docker load -i "$EXPORT_DIR/$image_name.tar"


    echo "Deploying $func_rel_path function..."
    /home/liwa/nuctl deploy --project-name cvat --path "$func_root" \
        --file "$func_config" --platform local
    /home/liwa/nuctl deploy --project-name cvat --path "$func_root" \
        --file "$func_config" --platform local \
        --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
        --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
        --platform-config '{"attributes": {"network": "cvat_cvat"}}'
done

/home/liwa/nuctl get function --platform local
