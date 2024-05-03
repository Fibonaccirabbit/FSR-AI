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

import CameraModel from '../model/msclassification/CameraModel';
import grantPermission from '../utlis/PermissionUtils';
import Logger from '../utlis/Logger';
import mobilenetV2 from 'libmobilenetV2.so';
import imageNet2012 from '../utlis/ImageNet2012Labels';
import ReturnBtn from '../component/ReturnBtn';
import TopBar from '../component/TopBar';

const TAG: string = '[FSR-AI-CLS]';

@Entry
@Component
struct CameraPage {
  private mXComponentController: XComponentController = new XComponentController();
  private surfaceId: string = '-1';
  private cameraModel: CameraModel = new CameraModel();
  @State modelId: number = 0;
  @State outputData: number = 0;
  @State fps: number = 0;

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
    Logger.info(TAG, 'onPageHide end');
  }

  onPageShow() {
    let context = getContext()
    let resourceManager = context.resourceManager;
    mobilenetV2.modelDeInit();
    try {
      this.modelId = mobilenetV2.modelInit(resourceManager);
      Logger.info(TAG, `modelId = ${this.modelId.toString()}`)
      if (this.modelId < 0) {
        mobilenetV2.modelDeInit();
      }
    } catch (err) {
      mobilenetV2.modelDeInit();
      Logger.info(TAG, `load model fail`);
    }
    this.surfaceId = this.mXComponentController.getXComponentSurfaceId();
    this.cameraModel.initCamera(this.surfaceId, this.modelId);
    Logger.info(TAG, `aboutToAppear,surfaceId=${this.surfaceId}`);
  }

  build() {
    Column() {
      TopBar({ text: '图像分类', media: $r('app.media.image_classification_icon') })
        .layoutWeight(1)
        .height('120px')
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
              this.cameraModel.onImageArrival((value: number, fps: number) => {
                this.outputData = value;
                this.fps = fps;
              })
            })

        }
        .height("960px")
        .width("60%")

        Column({ space: 20 }) {
          Text("mobilenetV2")
            .fontSize(52)
            .textAlign(TextAlign.Center)
            .backgroundColor('#F1F3F5')
            .width("70%")
            .height("22%")
          Text(this.fps.toString() + " img/s")
            .fontSize(48)
            .textAlign(TextAlign.Center)
          Text(imageNet2012[this.outputData])
            .fontSize(48)
            .textAlign(TextAlign.Center)
          ReturnBtn()
        }
        .align(Alignment.Top)
        .width("40%")
      }
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