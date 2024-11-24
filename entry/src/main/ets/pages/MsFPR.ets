

import CameraModel from '../model/reconstruction/CameraModel';
import grantPermission from '../utlis/PermissionUtils';
import Logger from '../utlis/Logger';
import fpr from 'libfpr.so';
import ReturnBtn from '../component/ReturnBtn';
import TopBar from '../component/TopBar';
import display from '@ohos.display';

const TAG: string = '[FSR-AI-CLS]';
@Entry
@Component
struct CameraPage {
  private mXComponentController: XComponentController = new XComponentController();
  private surfaceId: string = '-1';
  private cameraModel: CameraModel = new CameraModel();
  @State modelId: number = 0;
  @State band_700: string = "";
  @State band_400: string = "";
  @State fps: number = 0;
  @State swidth: string = "400px"
  @State sheight: string = "240px"
  private w :number =400;
  private h :number =240;

  async aboutToAppear() {
    try {
      await grantPermission();
      Logger.info(TAG, `Permission success`);
    } catch (err) {
      Logger.info(TAG, `Permission fail!`);
    }
    let displayClass: display.Display | null = null;
    try {
      displayClass = display.getDefaultDisplaySync();
      this.w = parseInt(displayClass.width.toString()) * 0.5;
      this.h = parseInt(displayClass.height.toString())  ;
      this.swidth = this.w.toString() + "px"
      this.sheight = this.h.toString() + "px"

      Logger.info(TAG, displayClass.width.toString() + " " + displayClass.height.toString());
    } catch (exception) {
      console.error('Failed to obtain the default display object. Code: ' + JSON.stringify(exception));
    }
  }

  onPageHide() {
    Logger.info(TAG, 'onPageHide begin');
    Logger.info(TAG, 'onPageHide end');
  }

  onPageShow() {
    let context = getContext()
    let resourceManager = context.resourceManager;
    fpr.modelDeInit();
    try {
      this.modelId = fpr.modelInit(resourceManager);
      Logger.info(TAG, `modelId = ${this.modelId.toString()}`)
      if (this.modelId < 0) {
        fpr.modelDeInit();
      }
    } catch (err) {
      fpr.modelDeInit();
      Logger.info(TAG, `load model fail`);
    }
    this.surfaceId = this.mXComponentController.getXComponentSurfaceId();
    this.cameraModel.initCamera(this.surfaceId, this.modelId);
    Logger.info(TAG, `aboutToAppear,surfaceId=${this.surfaceId}`);
  }

  build() {
    Column() {
      TopBar({ text: '高光譜重建', media: $r('app.media.image_classification_icon') })
        .layoutWeight(1)
        .height('30%')
      Row() {
        Column() {
          XComponent({
            id: 'componentId',
            type: 'surface',
            controller: this.mXComponentController
          })
            .onLoad(async () => {
              Logger.info(TAG, 'onLoad is called');
              this.mXComponentController.setXComponentSurfaceSize({ surfaceWidth: 1080, surfaceHeight: 1920 });
              this.surfaceId = this.mXComponentController.getXComponentSurfaceId();
              Logger.info(TAG, `onLoad surfaceId: ${this.surfaceId}`);
              this.cameraModel.initCamera(this.surfaceId, this.modelId);
              this.cameraModel.onImageArrival((base64: string, fps: number) => {
                let images = base64.split("@")
                this.band_700 = images[1];
                this.band_400 = images[0]
                this.fps = fps;
              })
            })

        }
        .height(this.sheight)
        .width(this.swidth)

        Column({ space: 20 }) {
          Text("RGB重构高光谱")
            .fontSize(23)
            .textAlign(TextAlign.Center)
            .backgroundColor('#F1F3F5')
            .width("70%")
            .height("12%")
          Image(this.band_700)
            .width("50%")
            .height("40%")
          Image(this.band_400)
            .width("50%")
            .height("40%")
          ReturnBtn()
        }
        .align(Alignment.Top)
        .width("30%")
      }
      .height("70%")
    }
    .justifyContent(FlexAlign.Center)
    .backgroundColor('#00fcfafa')
    .width('100%')
    .height('100%')
  }

  async aboutToDisappear() {
    await this.cameraModel.cameraRelease();
  }
}