var final_score = 0;
var final_level = "bad";
function ratio(score = 0.0) //score是0-1之间的两位小数
{        
    var dom = document.getElementById('ratio-container');
    var myChart = echarts.init(dom, null, {
      renderer: 'canvas',
      useDirtyRect: false
    });
    var app = {};
    
    var option;

    option = {
  series: [
    {
      type: 'gauge',
      startAngle: 180,
      endAngle: 0,
      center: ['50%', '75%'],
      radius: '90%',
      min: 0,
      max: 1,
      splitNumber: 8,
      axisLine: {
        lineStyle: {
          width: 6,
          color: [
            [0.25, '#FF6E76'],
            [0.5, '#FDDD60'],
            [0.75, '#58D9F9'],
            [1, '#07e162']
          ]
        }
      },
      pointer: {
        icon: 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
        length: '12%',
        width: 20,
        offsetCenter: [0, '-60%'],
        itemStyle: {
          color: 'auto'
        }
      },
      axisTick: {
        length: 12,
        lineStyle: {
          color: 'auto',
          width: 2
        }
      },
      splitLine: {
        length: 20,
        lineStyle: {
          color: 'auto',
          width: 5
        }
      },
      axisLabel: {
        color: '#464646',
        fontSize: 20,
        distance: -60,
        rotate: 'tangential',
        formatter: function (value) {
          if (value === 0.875) {
            return '优';
          } else if (value === 0.625) {
            return '良';
          } else if (value === 0.375) {
            return '中';
          } else if (value === 0.125) {
            return '差';
          }
          return '';
        }
      },
      title: {
        offsetCenter: [0, '-10%'],
        fontSize: 20
      },
      detail: {
        fontSize: 30,
        offsetCenter: [0, '-35%'],
        valueAnimation: true,
        formatter: function (value) {
          return Math.round(value * 100) + '';
        },
        color: 'inherit'
      },
      data: [
        {
          value: score,
          name: '安全性总评'
        }
      ]
    }
  ]
};

    if (option && typeof option === 'object') {
      myChart.setOption(option);
    }

    window.addEventListener('resize', myChart.resize);
}
  
function silder(score, score_value){
  const scoreInput = document.getElementById(score);
  const scoreValueDisplay = document.getElementById(score_value);

  // 监听滑块值的变化
  scoreInput.addEventListener('input', function() {
    // 获取滑块的值
    const score = scoreInput.value;
    // 更新显示选择的值
    scoreValueDisplay.textContent = score;
  });
}
function openFileManager() {
  // 触发文件选择框的点击事件
  var fileInput = document.getElementById('file-input');
  // 添加change事件监听器
  fileInput.addEventListener('change', function(event) {
    // 获取用户选择的文件
    var selectedFile = event.target.files[0];
    var file = selectedFile.name.split('.').slice(0, -1).join('.');
    var f = document.getElementById("model");
    f.innerHTML = file
    // 处理用户选择的文件
    console.log('用户选择的文件:', selectedFile);
    
    // 这里你可以进一步处理用户选择的文件，比如读取文件内容等操作
    model_file = selectedFile;
});

// 触发文件选择框的点击事件
fileInput.click();
}

function start_detect(){
  var data = {
    "model" : document.getElementById("model").innerText,  //目前只上传模型名，后端去特定路径下寻找模型。之后需要用户上传模型文件，后端接收并评测
    "device" : document.getElementById("device").value,
    "batch_size" :document.getElementById("batch_size_value").innerText, 
    "backdoor_method":document.getElementById("bd_method").value,
    "poison_epoch":document.getElementById("epoch_value").innerText, 
    "poison_batch_size":document.getElementById("bsize").value,
    "poison_method":document.getElementById("pmethod").value,
    "dataset":document.getElementById("dataset").value
  }
  show_load(1, true);
  show_load(2, true);
  show_load(3, true);

fetch('/detect', {
  method: 'POST',
  headers: {
      'Content-Type': 'application/json'
  },
  body: JSON.stringify(data)
})
.then(response => response.json())
.then(handleResponse)
.catch(error => {
  console.error('发生错误:', error);
});
}

function handleResponse(data) {
  console.log('收到来自服务器的响应:', data);
  show_load(1, false);
  show_load(2, false);
  show_load(3, false);
  // 这里可以对服务器的响应进行进一步处理，展示测评结果
  var scores = []
  for (var key in data) {
    if (key!='final_score') { 
      scores.push(data[key]);
    }
    else{
      final_score = data[key]/100;
      if (final_score>=0.875)
      {
        final_level = "excellent";
      }
      else if(final_score>=0.625 && final_score<0.875)
      {
        final_level = "good";
      }
      else if(final_score>=0.375 && final_score<0.625)
      {
        final_level = "middle";
      }
      else{
        final_level = "bad";
      }
      ratio(final_score);
    }
  }
  result(scores);
}

function handleError(error) {
  console.error('发生错误:', error);
  // 这里可以处理发生的错误
  alert("检测失败，请重新检测！")
}

function show_load(loadid,sign){
  if (loadid==1)
  {
    var p = document.getElementById('loading1');
  }
  else if (loadid==2)
  {
    var p = document.getElementById('loading2');
  }
  else if (loadid==3)
  {
    var p = document.getElementById('loading3');
  }
  if (sign==true){
    p.style.width = '100%';
    p.style.height = '100%';
    p.style.fontSize = '35px';
    p.style.display = 'flex';
    p.style.marginTop = '16vh';
    p.style.justifyContent = 'space-between';
  }
  else {
    p.style.display = 'none';
  } 
}

function result(scores){  //显示各项指标
  var values = document.getElementsByClassName("value");
  if (scores.length>0){
    for (var i=0; i<values.length; i++)
  {
    values[i].innerHTML = scores[i];
  }
  }
  else{
    for (var i=0; i<values.length; i++)
  {
    values[i].innerHTML = '—';
  }
  }
  
}
 
function report(){  //生成检测报告
  const doc = new jsPDF();
  const table = document.getElementById("result_table");
  const tableRows = table.rows;
  
  // Add title
  var model_name = document.getElementById("model").innerText;
  doc.text(`Model <${model_name}> Evaluate Result\n\n\nFinal_Score:${final_score}  Security_Level:${final_level}`, 10, 10);
  // var metrics = [
  //   ["CACC", "ASR"] ,
  //   ["MR-TA", "ACAC"] ,
  //   ["ACTC", "NTE"] ,
  //   ["ALD-p", "AQT"] ,
  //   ["CCV", "CAV"] ,
  //   ["COS", "RGB"] ,
  //   ["RIC", "T-std"] ,
  //   ["T-size", "CC"] 
  // ]
  // var values = [
  //   ["CACC", "ASR"] ,
  //   ["MR-TA", "ACAC"] ,
  //   ["ACTC", "NTE"] ,
  //   ["ALD-p", "AQT"] ,
  //   ["CCV", "CAV"] ,
  //   ["COS", "RGB"] ,
  //   ["RIC", "T-std"] ,
  //   ["T-size", "CC"] 
  // ]
  // for (let i = 1; i < tableRows.length; i++) {
  //   const cells = tableRows[i].cells;
  //   for (let j = 0; j < cells.length/2; j++) {
  //     var v = cells[j*2+1].innerText;
  //     values[i][j] = v;
  //   }
  // }
  // // Add table data
  // for (let i = 1; i < tableRows.length; i++) {
  //   const rowData = [];
  //   for (let j = 0; j < cells.length/2; j++) {
  //     rowData.push("    ")
  //     rowData.push(`${metrics[i-1][j]}:${values[i-1][j]}`);  
  //   }
  //   doc.text(10, 20 + (i * 10), rowData.join("  \n  "));
  // }
  // Save the PDF
  doc.save("table_data.pdf");
}