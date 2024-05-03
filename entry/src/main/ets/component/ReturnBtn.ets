import router from '@ohos.router';
@Extend(Button) function ReturnBtnFancy(con: Boolean = true) {
  .fontColor(con ? "#E84026" : "#0A59F7")
  .backgroundColor("#F1F3F5")
  .width(120)
  .height(40)
  .border({ width: 2, color: "#254FF7" })

}

@Component
struct ReturnBtn{

  build(){
    Column(){
      Button("退出")
        .ReturnBtnFancy()
        .onClick(()=>{
          router.back();
        })
    }
  }

}

export default ReturnBtn;