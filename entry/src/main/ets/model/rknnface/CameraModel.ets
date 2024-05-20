

import camera from '@ohos.multimedia.camera';
import image from '@ohos.multimedia.image';
import Logger from '../../utlis/Logger';
import { BusinessError, Callback } from '@ohos.base';
import emitter from "@ohos.events.emitter";
import rknnSocket from '../rknnsocket/rknnSokcet';
import common from '@ohos.app.ability.common';
import fs from '@ohos.file.fs';
import socket from '@ohos.net.socket';

import systemDateTime from '@ohos.systemDateTime';
// 定义一个eventId为1的事件，事件优先级为Low
let event: emitter.InnerEvent = {
  eventId: 1,
  priority: emitter.EventPriority.HIGH
};

interface CameraInfo {
  width: number;
  height: number;
}


interface ObjectDesc {
  left?: number;
  top?: number;
  right?: number;
  bottom?: number;
  prop?: number;
  name?: string;
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
  private socket: rknnSocket;

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
          const readBuffer: ArrayBuffer = new ArrayBuffer(pixelMap.getPixelBytesNumber()); //1382400
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
  onImageArrival(callback: (value: number, fps: number) => void): void {
    this.receiver.on('imageArrival', () => {

      this.receiver.readNextImage((err: BusinessError, nextImage: image.Image) => {
        if (err || nextImage === undefined) {
          return;
        }

        nextImage.getComponent(image.ComponentType.JPEG, (err: BusinessError, imgComponent: image.Component) => {
          if (err || imgComponent === undefined) {
            return;
          }

          let buffer: ArrayBuffer;
          if (imgComponent.byteBuffer as ArrayBuffer) {
            buffer = imgComponent.byteBuffer;
          } else {
            return;
          }

          // 如果每张img都处理，app会卡并且画框有滞后性能为13帧；每2张img推理1张会流畅很多，性能可达到12帧
          // 利用napi异步工作队列，最多有4个线程并行
          if ((number % 12) == 0) {
            // this.savePictureSand(buffer, nextImage);
            this.socket.sendImage(buffer);
            callback(1, 1);
          }
          number++;
          nextImage.release();
        })
      })

    })
  }

  /**
   * 初始化相机
   * @param surfaceId
   */
  async initCamera(surfaceId: string, modelId: number, socket: rknnSocket): Promise<void> {
    // this.modelId = modelId;
    this.socket = socket;
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
    Logger.info(TAG, `previewProfile = ${JSON.stringify(this.capability.previewProfiles)}`)
    let imageReceiveSurfaceId: string = await this.receiver.getReceivingSurfaceId();
    this.previewOutput = this.cameraMgr.createPreviewOutput(previewProfile, surfaceId);
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
    this.socket.close();
  }

  async getFileName() {
    let time;
    try {
      time = await systemDateTime.getCurrentTime()

    } catch (e) {
      console.info(`YZJ  Failed to get currentTime. message:${e.message}, code:${e.code}`);
    }
    if (time == null) {
      time = "Images"
    }
    return time + ".jpg"

  }

  async savePictureSand(buffer: ArrayBuffer, img: image.Image) {
    console.info("YZJ " + buffer.byteLength)
    let context = getContext(this) as common.UIAbilityContext;
    let cacheDir = context.cacheDir;
    let fileName = await this.getFileName();
    let filePath = cacheDir + "/" + fileName;
    let ss = fs.createStreamSync(filePath, "w+");
    console.info("YZJ 准备开始保存")
    ss.write(buffer, (err, bytesWritten) => {
      if (err) {
        console.info("YZJ write stream failed with error message: " + err.message + ", error code: " + err.code);
      } else {
        if (bytesWritten) {
          console.info("YZJ write succeed and size is:" + bytesWritten);
          img.release()
        }
      }
    });

  }
}
