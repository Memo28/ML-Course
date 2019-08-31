const classifier = knnClassifier.create();

let net;
const webcamElement = document.getElementById('webcam');


//Canvas load
var canv = document.getElementById("canvas_space");
var paper = canv.getContext("2d");
var x = 200;
var y = 200;
draw("red", x - 1, y - 1, x + 1, y - 1, paper);


async function realtime() {
	await load();
	//Cargamos la camara
	await setupWebcam();


	//Leemos las imagenes cargadas
	const addExample = classId => {
		const activation = net.infer(webcamElement, 'conv_preds');

		classifier.addExample(activation, classId);
	};

	document.getElementById('move-up').addEventListener('click', () => addExample(0));
	document.getElementById('move-down').addEventListener('click', () => addExample(1));
	document.getElementById('move-left').addEventListener('click', () => addExample(2));
	document.getElementById('move-right').addEventListener('click', () => addExample(3));


	while (true) {

		if (classifier.getNumClasses() > 0) {
			const activation = net.infer(webcamElement, 'conv_preds');


			const result = await classifier.predictClass(activation);

			const classes = ['UP', 'DOWN', 'LEFT', 'RIGHT'];
			document.getElementById('display').innerText = `Move To : prediction : ${classes[result.classIndex]}\n  probability : ${result.confidences[result.classIndex]}`;

			await moveInCanvas(classes[result.classIndex]);

		}
		await tf.nextFrame();
	}

}


async function setupWebcam() {
	return new Promise((resolve, reject) => {
		const navigatorAny = navigator;

		navigator.getUserMedia = navigator.getUserMedia || navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia || navigatorAny.msGetUserMedia;

		if (navigator.getUserMedia) {
			navigator.getUserMedia({ video: true },
				stream => {
					webcamElement.srcObject = stream;
					webcamElement.addEventListener('loadeddata', () => resolve(), false);
				},
				error => reject());
		} else {
			reject();
		}
	});
}


async function load() {
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



	result.forEach((item) => {
		let li = document.createElement('li');
		ul.appendChild(li);
		li.innerHTML += "Class Name: " + item.className + " Probability : " + item.probability;
	});

	console.log(result);


}


async function moveInCanvas(position) {
	var movement = 1;
	switch (position) {
		case 'UP':
			//A little pause
			draw("red", x, y, x, y - movement, paper);
			y = y - movement;
			setTimeout(() => { }, 10000);
			console.log('UP');
			break;
		case 'DOWN':
			draw("red", x, y, x, y + movement, paper);
			y = y + movement;
			setTimeout(() => { }, 10000);
			console.log('DOWN');
			break;
		case 'LEFT':
			draw("red", x, y, x - movement, y, paper);
			x = x - movement;
			setTimeout(() => { }, 10000);
			console.log('LEFT');
			break;
		case 'RIGHT':
			draw("red", x, y, x + movement, y, paper);
			x = x + movement;
			setTimeout(() => { }, 10000);
			console.log('RIGHT');
			break;
		default:
			setTimeout(() => { }, 10000);
			break;
	}
}



function draw(color, xstart, ystart, xfinal, yfinal, canv) {
	canv.beginPath();
	canv.strokeStyle = color;
	canv.lineWidth = 3;
	canv.moveTo(xstart, ystart);
	canv.lineTo(xfinal, yfinal);
	canv.stroke();
	canv.closePath();
}

// app()