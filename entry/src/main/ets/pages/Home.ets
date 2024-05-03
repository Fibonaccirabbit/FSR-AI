import router from '@ohos.router'
import abilityAccessCtrl from '@ohos.abilityAccessCtrl'
import TopBar from '../component/TopBar'
class functionButton{
  name: string
  media: ResourceStr
  url: string

  constructor(name: string, media: ResourceStr, url?: string) {
    this.name = name
    this.media = media
    this.url = url
  }
}

@Entry
@Component
struct Home {
  myArr: functionButton[] = [
    new functionButton('目标检测', $r('app.media.obj_detection_icon'),"pages/RknnYolo"),
    new functionButton('图像分类', $r('app.media.image_classification_icon'),"pages/MsMobilenetV2"),
    new functionButton('图像分割', $r('app.media.image_segmentation_icon'),"pages/RknnSeg"),
    new functionButton('人脸识别', $r('app.media.face_icon'),"pages/RknnRetinaFace"),
    new functionButton('文字识别', $r('app.media.text_detection_icon'),"pages/RknnRetinaFace"),
    new functionButton('手势识别', $r('app.media.text_detection_icon'),"pages/RknnHandpose"),
    new functionButton('测试页面', $r('app.media.text_detection_icon'),"pages/Index"),
    new functionButton('深度估計', $r('app.media.text_detection_icon'),"pages/MsDepthAnything"),
    new functionButton('高光譜重建', $r('app.media.text_detection_icon'),"pages/MsFPR"),
  ]
  async aboutToAppear() {
    let AtManager = abilityAccessCtrl.createAtManager();
    AtManager.requestPermissionsFromUser(getContext(this), ['ohos.permission.READ_MEDIA', 'ohos.permission.MEDIA_LOCATION', "ohos.permission.LOCATION", "ohos.permission.CAMERA", "ohos.permission.INTERNET"])
  }
  build() {
    Column() {
      TopBar({ text: 'FSR-AI算法包', media: $r('app.media.AI') })
        .layoutWeight(1)
        .height('20%')
      Flex({ justifyContent: FlexAlign.SpaceAround, wrap: FlexWrap.Wrap }) {
        ForEach(this.myArr, (item: functionButton, index) => {
          Column({space:5}) {
            Image(item.media)
              .width(40)
            Text(item.name)
          }
          .onClick(() => {
            router.pushUrl({ url: item.url })
          })
          .margin(35)
          .backgroundColor(Color.White)
          .padding(10)
          .borderRadius(15)
          .shadow({ radius: 6, color: '#1F000000', offsetX: 2, offsetY: 4 })
          .width(90)
          .height(90)
        })
      }
      .padding(15)
      .height('100%')
      .layoutWeight(7)
    }
    .backgroundColor('#f1f2f3')
    .width('100%')
    .height('100%')
  }
}

