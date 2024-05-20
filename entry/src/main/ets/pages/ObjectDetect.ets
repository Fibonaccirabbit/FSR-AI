

import CameraModel from '../model/objectdetect/CameraModel';
import grantPermission from '../utlis/PermissionUtils';
import Logger from '../utlis/Logger';
import util from '@ohos.util';
import { BusinessError } from '@ohos.base';
import prompt from '@ohos.prompt';
import image from '@ohos.multimedia.image';
import fs from '@ohos.file.fs';

import ssd300 from 'libssd300.so';
import imageNet2012 from '../utlis/ImageNet2012Labels';

const TAG: string = '[FSR-AI-OD]';

interface ObjectDesc {
  left?: number;
  top?: number;
  right?: number;
  bottom?: number;
  prop?: number;
  name?: string;
}

@Entry
@Component
struct CameraPage {
  private mXComponentController: XComponentController = new XComponentController();
  private surfaceId: string = '-1';
  private cameraModel: CameraModel = new CameraModel();
  @State imageThumbnail: string = '';
  @State videoThumbnail: Resource | undefined = undefined;
  @State isRecording: boolean = false;
  @State textMove: number = 45;
  @State isPointShow: boolean = true;
  @State isTitleShow: boolean = true;
  @State timeShow: boolean = false;
  @State format: string = 'mm:ss';
  @State message: string = 'Hello World';
  @State widthCanvas: number = 100;
  @State heightCanvas: number = 100;
  @State p1x: number = 0;
  @State p1y: number = 0;
  @State p2x: number = 0;
  @State p2y: number = 0;
  @State p3x: number = 0;
  @State p3y: number = 0;
  @State p4x: number = 0;
  @State p4y: number = 0;
  @State boxQuan: number = 0;
  @State colorSet: Color[] = [Color.Red, Color.Green, Color.Yellow, Color.Blue];
  @State indexSet: number[] = [1, 2, 3];
  @State imageHeight: number = 0;
  @State imageWidth: number = 0;
  @State dataSize: number = 0;
  @State result: number = 0;
  @State resultTime: number = 0;
  @State modelId: number = 0;
  @State outputData: number = 0;
  @State imageNumber: number = 0;
  @State numX: number = 0.0;
  @State conf: string = '0';
  @State fps: number = 0;
  private context: Context = getContext(this);
  private textInputControllerX: TextInputController = new TextInputController();
  @State pointData: ObjectDesc[] = [];

  async aboutToAppear() {
    try {
      await grantPermission();
      Logger.info(TAG, `Permission success`);
    } catch (err) {
      Logger.info(TAG, `Permission fail!`);
    }
  }

  onPageHide() {
    Logger.info(TAG, 'onPageHide begin');
    if (this.isRecording) {
      Logger.info(TAG, 'stopVideo begin');
      this.cameraModel.cameraRelease();

    }
    Logger.info(TAG, 'onPageHide end');
  }

  onPageShow() {
    let context = getContext()
    let resourceManager = context.resourceManager;
    ssd300.modelDeInit();
    try {
      this.modelId = ssd300.modelInit(resourceManager);
      Logger.info(TAG, `modelId = ${this.modelId.toString()}`)
      if (this.modelId < 0) {
        ssd300.modelDeInit();
      }
    } catch (err) {
      ssd300.modelDeInit();
      Logger.info(TAG, `load model fail`);
    }
    this.surfaceId = this.mXComponentController.getXComponentSurfaceId();
    this.cameraModel.initCamera(this.surfaceId, this.modelId);
    Logger.info(TAG, `aboutToAppear,surfaceId=${this.surfaceId}`);
  }

  build() {
    Column() {
      Row() {
        Stack() {
          ForEach(this.pointData, (item: ObjectDesc, index) => {
            Column() {
              Text(item.name)
                .fontSize(24)
                .fontColor(this.colorSet[index % 4])
              Text(String(item.prop * 100).substring(0, 4) + '%')
                .fontSize(12)
                .fontColor(this.colorSet[index % 4])
            }
            .position({ x: px2vp(Math.abs(Math.floor(1280 - item.bottom))), y: px2vp(Math.floor(item.left)) })
            .width(px2vp((Math.abs(Math.floor(item.bottom - item.top)))))
            .height(px2vp((Math.abs(Math.floor(item.right - item.left)))))
            .borderWidth(2)
            .borderColor(this.colorSet[index % 4])
            .zIndex(index + 1)
          })
          Column() {
            XComponent({
              id: 'componentId',
              type: 'surface',
              controller: this.mXComponentController
            })
              .onLoad(async () => {
                Logger.info(TAG, 'onLoad is called');
                this.mXComponentController.setXComponentSurfaceSize({ surfaceWidth: 1280, surfaceHeight: 960 });
                this.surfaceId = this.mXComponentController.getXComponentSurfaceId();
                Logger.info(TAG, `onLoad surfaceId: ${this.surfaceId}`);
                this.cameraModel.initCamera(this.surfaceId, this.modelId);
                this.cameraModel.onImageArrival((value: ObjectDesc[]) => {
                  this.pointData = value;
                  Logger.info(TAG,JSON.stringify(value))
                })
              })
              .width('1280px')
              .height('960px')
          }.zIndex(0)
        }
        .width('1280px')
        .height('960px')

        // Column({ space: 20 }) {
        //   Text("SSD300")
        //     .fontSize(52)
        //     .textAlign(TextAlign.Center)
        //     .backgroundColor('#F1F3F5')
        //     .width("70%")
        //     .height("22%")
        //   // Text(this.fps.toString() + " img/s")
        //   //   .fontSize(48)
        //   //   .textAlign(TextAlign.Center)
        //   // Text(imageNet2012[this.outputData])
        //   //   .fontSize(48)
        //   //   .textAlign(TextAlign.Center)
        // }
        // .align(Alignment.Top)

      }
    }
    .justifyContent(FlexAlign.Center)
    .backgroundColor('#00fcfafa')
    .width('100%')
    .height('100%')
  }

  uint8ArrayToString(uint8Array: Uint8Array): string {
    let textDecoder = util.TextDecoder.create('utf-8');
    return textDecoder.decodeWithStream(uint8Array);
  }

  uint8ArrayToStringTest(u8a: Uint8Array) {
    let dataStr = "";
    for (let i = 0; i < u8a.length; i++) {
      dataStr += String.fromCharCode(u8a[i])
    }
    return dataStr;
  }

  async aboutToDisappear() {
    await this.cameraModel.cameraRelease();
  }
}