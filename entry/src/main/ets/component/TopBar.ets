
@Component
struct TopBar {
  media: ResourceStr
  mediaSize: number = 35
  text: string
  fontSize: number = 25
  setBool: boolean = true
  setSize: number = 35
  onClickFun: Function

  build() {
    Row({space: 10}) {
      Image(this.media)
        .width(this.mediaSize)
      Text(this.text)
        .fontSize(this.fontSize)
        .fontColor(Color.White)
      Blank()
      if (this.setBool) {
        Image($r('app.media.set'))
          .width(this.setSize)
          .onClick(() => {
            this.onClickFun
          })
      }
    }
    .padding(15)
    .justifyContent(FlexAlign.SpaceBetween)
    .width('100%')
    .backgroundColor('#ff021b39')
  }
}

export default TopBar