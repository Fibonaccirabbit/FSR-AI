/*
 * Copyright 2023 Unionman Technology Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import camera from '@ohos.multimedia.camera';
import image from '@ohos.multimedia.image';
import mediaLibrary from '@ohos.multimedia.mediaLibrary';
import Logger from '../../utlis/Logger';
import { BusinessError, Callback } from '@ohos.base';
import fpr from 'libfpr.so';
import emitter from "@ohos.events.emitter";
import performanceMonitor from '@ohos.arkui.performanceMonitor';

// 定义一个eventId为1的事件，事件优先级为Low
let event: emitter.InnerEvent = {
  eventId: 1,
  priority: emitter.EventPriority.HIGH
};
interface CameraInfo {
  width: number;
  height: number;
}

interface InferResult {
  count?: number;
  objects?: ObjectDesc[];
}

interface ObjectDesc {
  left?: number;
  top?: number;
  right?: number;
  bottom?: number;
  prop?: number;
  name?: string;
}

interface ClsInferResult {
  count?: number;
  objects?: ClsDesc[];
}

interface ClsDesc {
  index:number;
  score:number;
}

interface PicDesc {
  width: number;
  height: number;
  dataSize: number;
}

let cameraWH: CameraInfo = {
  width: 1080,
  height: 1920,
};

let number: number = 0;

const TAG = '[FSR-AI]';

export default class CameraService {
  private cameraMgr: camera.CameraManager = undefined;
  private camerasArray: Array<camera.CameraDevice> = undefined;
  private cameraInput: camera.CameraInput = undefined;
  private previewOutput: camera.PreviewOutput = undefined;
  private previewOutput2: camera.PreviewOutput = undefined;
  private photoOutPut: camera.PhotoOutput = undefined;
  private capSession: camera.CaptureSession = undefined;
  private videoOutput: camera.VideoOutput = undefined;
  private capability: camera.CameraOutputCapability = undefined;
  private receiver: image.ImageReceiver = undefined;
  private context: Context = getContext(this);
  private modelId: number = -1;
  private isFirst: boolean = true;

  constructor() {
    this.receiver = image.createImageReceiver(
      cameraWH.width,
      cameraWH.height,
      image.ImageFormat.JPEG,
      1
    );
  }

  setImageTest(imageSet: Resource): Uint8Array {
    let imageBuffer: Uint8Array;
    try {
      imageBuffer = this.context.resourceManager.getMediaContentSync(imageSet.id); // 默认屏幕密度
    } catch (error) {
      let code = (error as BusinessError).code;
      let message = (error as BusinessError).message;
      console.error(`getMediaContentSync failed, error code: ${code}, message: ${message}.`);
    }
    Logger.info(TAG, `imageBuffer = ${imageBuffer}`);
    return imageBuffer;
  }

  arrayBufferToPixelMap(imageBuffer: ArrayBuffer) {
    const imageSourceApi: image.ImageSource = image.createImageSource(imageBuffer);
    imageSourceApi.getImageInfo(0)
      .then((imageInfo: image.ImageInfo) => {
        Logger.info(TAG, `Succeeded in obtaining the image information. and width = ${imageInfo.size.width} , height = ${imageInfo.size.height} `);
        imageSourceApi.createPixelMap().then((pixelMap: image.PixelMap) => {
          Logger.info(TAG, `Succeeded in creating pixelMap object through image decoding parameters.`);
          const readBuffer: ArrayBuffer = new ArrayBuffer(pixelMap.getPixelBytesNumber());//1382400
          pixelMap.readPixelsToBuffer(readBuffer).then(() => {
            Logger.info(TAG, 'Succeeded in reading image pixel data.'); //符合条件则进入

            // Logger.info(TAG, `pixelMapTest = ${uint8Array}`);
          }).catch((error: BusinessError) => {
            Logger.info(TAG, 'Failed to read image pixel data.'); //不符合条件则进入
          })
        }).catch((error: BusinessError) => {
          Logger.error(TAG, `Failed to create pixelMap and error = ${JSON.stringify(error)}`);
        })
      }).catch((error: BusinessError) => {
      Logger.error('Failed to obtain the image information.');
    })
  }

  /**
   * 创建ImageReceiver组件Surface
   * @returns
   */
  async getImageReceiverSurfaceId(): Promise<string | undefined> {
    this.receiver = image.createImageReceiver(1080, 1920, 4, 8);
    console.info('before ImageReceiver check');
    let ImageReceiverSurfaceId: string | undefined = undefined;
    if (this.receiver !== undefined) {
      console.info('ImageReceiver is ok');
      let ImageReceiverSurfaceId: string = await this.receiver.getReceivingSurfaceId();
      console.info(`ImageReceived id: ${ImageReceiverSurfaceId}`);
    } else {
      console.info('ImageReceiver is not ok');
    }
    return ImageReceiverSurfaceId;
  }

  /**
   * 通过ImageReceiver实时获取预览图像。
   * @param receiver
   */
  onImageArrival(callback: Callback<ArrayBuffer>): void {
    this.receiver.on('imageArrival', () => {
      //setInterval(() => {
      this.receiver.readNextImage((err: BusinessError, nextImage: image.Image) => {
        if (err || nextImage === undefined) {
          return;
        }

        nextImage.getComponent(image.ComponentType.JPEG,  (err: BusinessError, imgComponent: image.Component) => {
          if (err || imgComponent === undefined) {
            return;
          }

          let buffer: ArrayBuffer;
          if (imgComponent.byteBuffer as ArrayBuffer) {
            buffer = imgComponent.byteBuffer;
          } else {
            return;
          }

          let picDesc: PicDesc = {
            width: 640,
            height: 480,
            dataSize: buffer.byteLength
          }
          // let inputBuffer = await this.test(buffer);
          let f32 = new Float32Array(640*480*4);
          f32.set(new Float32Array(new Uint8Array(buffer)))
          console.info(TAG+f32[f32.length-1]+" "+buffer.byteLength.toString())
          // 如果每张img都处理，app会卡并且画框有滞后性能为13帧；每2张img推理1张会流畅很多，性能可达到12帧
          // 利用napi异步工作队列，最多有4个线程并行
          if ((number % 12) == 0 ) {
              // @ts-ignore
            Logger.info(TAG, `================Process pic number = ${number}`);

            console.info(TAG+f32.byteLength.toString())
            let value =fpr.modelInference(f32.buffer);
            callback(value);
          }
          number ++;
          nextImage.release();
        })
      })
      //}, 40)
    })
  }

  /**
   * 初始化相机
   * @param surfaceId
   */
  async initCamera(surfaceId: string, modelId: number): Promise<void> {
    this.modelId = modelId;
    await this.cameraRelease();
    Logger.info(TAG, 'getCameraManager begin');
    try {
      this.cameraMgr = camera.getCameraManager(this.context);
    } catch (e) {
      Logger.info(TAG, `getCameraManager catch message:${JSON.stringify(e.message)}`);
    }
    Logger.info(TAG, `getCameraManager ${JSON.stringify(this.cameraMgr)}`);
    this.camerasArray = this.cameraMgr.getSupportedCameras();
    Logger.info(TAG, `get cameras ${this.camerasArray.length}`);
    if (this.camerasArray.length === 0) {
      Logger.info(TAG, 'cannot get cameras');
      return;
    }

    let mCamera = this.camerasArray[0];
    Logger.info(TAG, `mCameraId = ${mCamera.cameraId} cameraPosition = ${mCamera.cameraPosition} cameraType = ${mCamera.cameraType} connectionType = ${mCamera.connectionType}`)
    this.cameraInput = this.cameraMgr.createCameraInput(mCamera);
    try {
      this.cameraInput.open().then(() => {
        Logger.info(TAG, 'Promise returned with camera opened.');
      }).catch((err: BusinessError) => {
        Logger.info(TAG, `err = ${JSON.stringify(err)}`)
      });
      Logger.info(TAG, 'Promise returned with camera opened.');
    } catch (err) {
      Logger.info(TAG, `Failed to open the camera. ${err.code}`);
    }
    Logger.info(TAG, 'createCameraInput');
    this.capability = this.cameraMgr.getSupportedOutputCapability(mCamera);
    let previewProfile = this.capability.previewProfiles[0];
    let previewProfile2 = this.capability.previewProfiles[0];
    Logger.info(TAG, `previewProfile = ${JSON.stringify(previewProfile)}`)
    this.previewOutput = this.cameraMgr.createPreviewOutput(previewProfile, surfaceId);
    let imageReceiveSurfaceId: string = await this.receiver.getReceivingSurfaceId();
    this.previewOutput2 = this.cameraMgr.createPreviewOutput(previewProfile2, imageReceiveSurfaceId);
    Logger.info(TAG, 'createPreviewOutput');
    try {
      this.capSession = this.cameraMgr.createCaptureSession();
    } catch (err) {
      Logger.info(TAG, `${JSON.stringify(err)}`)
    }
    Logger.info(TAG, 'createCaptureSession');

    this.capSession.beginConfig();
    Logger.info(TAG, 'beginConfig');
    this.capSession.addInput(this.cameraInput);
    this.capSession.addOutput(this.previewOutput);
    this.capSession.addOutput(this.previewOutput2)
    try {
      await this.capSession.commitConfig();
      Logger.info(TAG, 'Promise returned to indicate the commit config success.');
    } catch (err) {
      Logger.error(TAG, 'Failed to commitConfig ' + err.code);
    }
    try {
      this.capSession.start()
      Logger.info(TAG, 'Promise returned to indicate the session start success.');
    }
    catch (err) {
      Logger.error(TAG, `Failed to start the session ${err.code}`);
    }
    Logger.info(TAG, 'captureSession start');
  }


  /**
   * 资源释放
   */
  async cameraRelease(): Promise<void> {
    Logger.info(TAG, 'releaseCamera');
    if (this.cameraInput) {
      await this.cameraInput.close();
    }
    if (this.previewOutput) {
      await this.previewOutput.release();
    }
    if (this.photoOutPut) {
      await this.photoOutPut.release();
    }
    if (this.videoOutput) {
      await this.videoOutput.release();
    }
    if (this.capSession) {
      await this.capSession.release();
    }
  }

  async test(inputBuffer){
    let sourceOptions: image.SourceOptions = {
      sourceDensity: 120,
      sourcePixelFormat: 3,
      sourceSize: { height: 224, width: 224 }
    };
    const imageSourceApi = image.createImageSource(inputBuffer);
    let decodingOptions: image.DecodingOptions = {
      sampleSize: 1,
      editable: true,
      desiredSize: { width: 224, height: 224 },
      desiredPixelFormat: 3,
    };
    let pixel: PixelMap = await imageSourceApi.createPixelMap(decodingOptions)
    // 场景一：将读取的整张图像像素数据结果写入ArrayBuffer中
    let readBuffer : ArrayBuffer = new ArrayBuffer(pixel.getPixelBytesNumber());
    await pixel.readPixelsToBuffer(readBuffer).then(() => {
      console.log(TAG + '=========无损 input image byte length: ' + readBuffer.byteLength)
      console.info(TAG + 'Succeeded in reading image pixel data.');
    }).catch((error: string) => {
      console.error(TAG + 'Failed to read image pixel data. And the error is: ' + JSON.stringify(error));
    })

    return readBuffer;
  }

}
