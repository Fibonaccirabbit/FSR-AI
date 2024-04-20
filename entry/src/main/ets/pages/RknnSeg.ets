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

import CameraModel from '../model/rknnseg/CameraModel';
import Logger from '../utlis/Logger';
import util from '@ohos.util';
import emitter from "@ohos.events.emitter";
import display from '@ohos.display';
import rknnSocket from '../model/rknnsocket/rknnSokcet';
import socket from '@ohos.net.socket';

const TAG: string = '[FSR-AI-RKNN]';

interface ObjectDesc {
  xmin?: number;
  ymin?: number;
  xmax?: number;
  ymax?: number;
  label?: string;
  prop?: number;
}
// 定义一个eventId为1的事件
let RknnSegEvent: emitter.InnerEvent = {
  eventId: 4
};

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
  @State xmin: number = 0;
  @State ymin: number = 0;
  @State xmax: number = 0;
  @State ymax: number = 0;
  @State pointData: ObjectDesc[] = [];
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
  @State base64: string = "";
  private scale_factor_x = 2;
  private scale_factor_y = 2;
  private camera_show_width = 1280;
  private camera_show_height = 1080;
  public socket: rknnSocket = new rknnSocket("127.0.0.1", 12600);

  aboutToAppear() {
    // 收到eventId为1的事件后执行该回调
    this.onMsg();
  }

  onMsg() {
    this.socket.bindLocal();
    this.socket.tcpServer.on('message', (value: SocketInfo) => {
      Logger.info(TAG, "on message");
      let buffer = value.message;
      let dataView = new DataView(buffer);
      let str = "";
      for (let i = 0; i < dataView.byteLength; ++i) {
        str += String.fromCharCode(dataView.getUint8(i));
      }
      this.base64 = str;
      Logger.info(TAG, "on connect received:" + str);
    });
  }

  onPageHide() {

  }

  onPageShow() {

  }

  build() {
    Column() {
      Row() {
        Stack() {
          Image(this.base64)

            .zIndex(2)
            .opacity(0.5)
            .objectFit(ImageFit.Cover)
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
              this.cameraModel.initCamera(this.surfaceId, this.modelId,this.socket);
              this.cameraModel.onImageArrival((value: number, fps: number) => {
              })
            })
        }
        .backgroundColor("#ffc63333")
        .alignSelf(ItemAlign.Start)
        .width("1280px")
        .height("960px")

        Column({ space: 20 }) {
          Text("rknn Yolov5")
            .fontSize(42)
            .textAlign(TextAlign.Center)
            .backgroundColor('#F1F3F5')
            .width("70%")
            .height("22%")
          Text(this.fps.toString() + " img/s")
            .fontSize(48)
            .textAlign(TextAlign.Center)

        }
        .width("640px")
        .align(Alignment.Top)

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

class SocketInfo {
  message: ArrayBuffer = new ArrayBuffer(1);
  remoteInfo: socket.SocketRemoteInfo = {} as socket.SocketRemoteInfo;
}