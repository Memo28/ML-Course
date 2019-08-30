const classifier = knnClassifier.create();

let net;
const webcamElement = document.getElementById('webcam');

async function realtime(){
	await load();
	//Cargamos la camara
	await setupWebcam();


	//Leemos las imagenes cargadas
	const addExample = classId =>{
		const activation = net.infer(webcamElement, 'conv_preds');

		classifier.addExample(activation, classId);
	};

  	document.getElementById('class-a').addEventListener('click', () => addExample(0));
	document.getElementById('class-b').addEventListener('click', () => addExample(1));
	document.getElementById('class-c').addEventListener('click', () => addExample(2));


	while(true){

		if(classifier.getNumClasses () > 0){
			const activation = net.infer(webcamElement, 'conv_preds');

			const result = await classifier.predictClass(activation);

			const classes = ['A', 'B', 'C'];
			document.getElementById('console').innerText = `prediction : ${classes[result.classIndex]}\n  probability : ${result.confidences[result.classIndex]}`;
		}

		await tf.nextFrame();
	}

}


async function setupWebcam(){
	return new Promise((resolve, reject) =>{
		const navigatorAny = navigator;

		navigator.getUserMedia = navigator.getUserMedia || navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia || navigatorAny.msGetUserMedia;

		if(navigator.getUserMedia){
			navigator.getUserMedia({video : true},
				stream =>{
					webcamElement.srcObject = stream;
					webcamElement.addEventListener('loadeddata', () => resolve(), false);
				},
				error => reject());
		}else{
			reject();
		}
	});
}


async function load(){
	console.log('Loading mobilenet...');

	//Load the model
	net = await mobilenet.load();
	console.log('Sucessfully loaded...');
}

async function classifierFunction() {

	await load();

	const imgEl = document.getElementById('img');
	const result = await net.classify(imgEl);
	document.getElementById('ul-list').innerHTML = "";
	let ul = document.getElementById('ul-list');



	result.forEach((item) =>{
		let li = document.createElement('li');
		ul.appendChild(li);
		li.innerHTML += "Class Name: " + item.className + " Probability : " + item.probability;
	});

	console.log(result);

	
}

// app()