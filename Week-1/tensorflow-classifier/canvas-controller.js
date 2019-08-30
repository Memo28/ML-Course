document.body.addEventListener('click', (e) => {
    if(e.target.id == 'nav-1'){
        //Change in tabs
        document.getElementById('nav-2').classList.remove('active')
        document.getElementById('nav-1').classList.add('active')

        //Display block
        document.getElementById('classifier').style.display ="block"
        document.getElementById('real-time-detection').style.display ="none"

    }else if(e.target.id == 'nav-2'){
        //Change in tabs
        document.getElementById('nav-1').classList.remove('active')
        document.getElementById('nav-2').classList.add('active')

        //Display block

        document.getElementById('classifier').style.display ="none"
        document.getElementById('real-time-detection').style.display ="block"
    }
});


function loadFile(event) {
    var output = document.getElementById('img');
    output.src = URL.createObjectURL(event.target.files[0]);
  };
