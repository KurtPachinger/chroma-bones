//document.getElementById('rendered-js').type = 'module';
//module globals in head
//r123+ skins mesh differently :(
import * as THREE from "https://unpkg.com/three@0.122.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.122.0/examples/jsm/controls/OrbitControls.js";
import { GUI } from "https://unpkg.com/three@0.122.0/examples/jsm/libs/dat.gui.module.js";
import { TessellateModifier } from "https://unpkg.com/three@0.122.0/examples/jsm/modifiers/TessellateModifier.js";
import { GLTFExporter } from "https://unpkg.com/three@0.122.0/examples/jsm/exporters/GLTFExporter.js";

//docs.opencv.org/trunk/d2/df0/tutorial_js_table_of_contents_imgproc.html
let utils = new Utils("errorMessage");
utils.loadOpenCv(() => {
  grabCut();
  //utils.createFileFromUrl('/haarcascade_frontalface_default.xml', 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml', faceDetect);
});

utils.addFileInputHandler("photo", "chroma");
utils.addFileInputHandler("mask", "mask");

utils.loadImageToCanvas = function (url, cavansId) {
  let img = document.getElementById(cavansId + "Img");
  img.crossOrigin = "anonymous";
  img.onload = function () {
    if (cavansId == "mask") {
      img.src = url;
      return true;
    }
    let context = touchup.getContext("2d");
    context.clearRect(0, 0, touchup.width, touchup.height);
    let canvas = document.getElementById(cavansId);
    let ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, img.width, img.height);
    grabCut();
  };
  img.src = url;
};

//settings
sto = {
  faces: [],
  contours: [],
  bones: [],
  boneIndexes: {},
  state: {
    animateBones: true
  },
  update: function (init) {
    if (init) {
      document.documentElement.className = "loading";
      if (scene) {
        for (let i = scene.children.length - 1; i >= 0; i--) {
          let ch = scene.children[i];
          if (ch.type === "Group") {
            scene.remove(ch);
          }
        }
      }
      sto.faces = [];
      sto.contours = [];
      sto.bones = [];
      sto.export3d = [];

      //UI disabled
      document.getElementsByTagName("fieldset")[0].disabled = true;
    }
    //src img normalized to 1 for 3d origin/precision, and quality resolution-dependent
    let src = document.getElementById("chromaImg");
    sto.width = src.width;
    sto.height = src.height;
    renderer && renderer.setSize(sto.width, sto.height);
  }
};

function faceDetect() {
  console.log("faceDetect");

  let src = cv.imread("chromaImg");

  let gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
  let faces = new cv.RectVector();
  let faceCascade = new cv.CascadeClassifier();
  // load pre-trained classifiers (face detect)
  faceCascade.load("haarcascade_frontalface_default.xml");
  let minSize = new cv.Size(sto.width / 8, sto.height / 8),
    maxSize = new cv.Size(sto.width * 2, sto.height * 2);

  //www.emgu.com/wiki/files/1.5.0.0/Help/html/e2278977-87ea-8fa9-b78e-0e52cfe6406a.htm
  faceCascade.detectMultiScale(
    gray,
    faces,
    1.05,
    9,
    cv.CASCADE_FIND_BIGGEST_OBJECT | cv.CASCADE_DO_ROUGH_SEARCH,
    minSize,
    maxSize
  );
  faceCascade.delete();

  let decal = cv.imread("maskImg");
  for (let i = 0; i < faces.size(); ++i) {
    let face = faces.get(i);
    let faceUnBorder = faces.get(i);

    faceUnBorder.rowColBind = [];
    sto.faces.push(faceUnBorder);

    let roiGray = gray.roi(face);
    let roiSrc = src.roi(face);
    let point1 = new cv.Point(face.x, face.y);
    let point2 = new cv.Point(face.x + face.width, face.y + face.height);
    //cv.rectangle(src, point1, point2, [255, 0, 0, 255]);

    //facemask
    let mskWH = [point2.x - point1.x, point2.y - point1.y];
    cv.resize(
      decal,
      decal,
      new cv.Size(mskWH[0], mskWH[1]),
      0,
      0,
      cv.INTER_NEAREST
    );
    let mskRoi = src.roi(new cv.Rect(point1.x, point1.y, mskWH[0], mskWH[1]));

    //mask alpha
    let alpha = new cv.Mat();
    cv.cvtColor(decal, alpha, cv.COLOR_BGR2GRAY);
    cv.threshold(alpha, alpha, 0, 255, cv.THRESH_BINARY);

    decal.copyTo(mskRoi, alpha);

    alpha.delete();
    roiGray.delete();
    roiSrc.delete();
  }
  decal.delete();
  faces.delete();
  gray.delete();

  cv.imshow("chroma", src);
  src.delete();

  grabCut();
}

async function grabCut() {
  let prog = new Promise((resolve, reject) => {
    //promotes css
    setTimeout(() => resolve("reset, loading/hi-res src"), 0);
    sto.update(true);
  });
  let result = await prog;

  console.log("grabCut");
  //todo: black/transparent are lost

  let src = cv.imread("chromaImg"); //skip faceDetect
  cv.cvtColor(src, src, cv.COLOR_RGBA2RGB, 0);
  let mask = new cv.Mat.zeros(src.size(), cv.CV_8UC1);
  const D = ((sto.width + sto.height) / 2) * 0.15;

  let GC = {
    //answers.opencv.org/question/132163/grabcut-mask-values/
    BGD: new cv.Scalar(cv.GC_BGD),
    FGD: new cv.Scalar(cv.GC_FGD),
    PR_BGD: new cv.Scalar(cv.GC_PR_BGD),
    PR_FGD: new cv.Scalar(cv.GC_PR_FGD),
    GreenScreen: function (i, j) {
      if (
        src.ucharPtr(i, j)[0] < 48 &&
        src.ucharPtr(i, j)[1] > 160 &&
        src.ucharPtr(i, j)[2] < 48
      ) {
        return true;
      }
    }
  };

  //helper rects
  let GC_PR = [new cv.Point(0, 0), new cv.Point(sto.width, sto.height)];
  cv.rectangle(mask, GC_PR[0], GC_PR[1], GC.PR_FGD, -1, 4, 0);
  cv.rectangle(mask, GC_PR[0], GC_PR[1], GC.PR_BGD, D * 2, 4, 0);
  //corners background
  cv.circle(mask, new cv.Point(0, 0), D * 2, GC.PR_BGD, -1, 4, 0);
  cv.circle(mask, new cv.Point(sto.width, 0), D * 2, GC.PR_BGD, -1, 4, 0);
  cv.circle(mask, new cv.Point(0, sto.height), D * 2, GC.PR_BGD, -1, 4, 0);
  cv.circle(
    mask,
    new cv.Point(sto.width, sto.height),
    D * 2,
    GC.PR_BGD,
    -1,
    4,
    0
  );

  let touchup = cv.imread("touchup");
  let dsize = new cv.Size(sto.width, sto.height);
  cv.resize(touchup, touchup, dsize, 0, 0, cv.INTER_AREA);

  //greenscreen samples
  for (let i = 0; i < src.rows; i += 3) {
    for (let j = 0; j < src.cols; j += 3) {
      //if touchup mask pixel >50% opaque, G channel value
      let point1 = new cv.Point(j - 1, i - 1);
      let point2 = new cv.Point(j + 1, i + 1);
      if (touchup.ucharPtr(i, j)[3] > 127) {
        if (touchup.ucharPtr(i, j)[1] < 64) {
          cv.rectangle(mask, point1, point2, GC.BGD, -1, cv.LINE_8, 0);
        } else if (touchup.ucharPtr(i, j)[1] > 192) {
          cv.rectangle(mask, point1, point2, GC.FGD, -1, cv.LINE_8, 0);
        }
      } else if (GC.GreenScreen(i, j)) {
        mask.ucharPtr(i, j)[0] = GC.PR_BGD;
        //cv.rectangle(mask, point1, point2, GC.PR_BGD, -1, cv.LINE_8, 0);
      }
    }
  }

  let faces = sto.faces;
  for (let i = 0; i < faces.length; ++i) {
    //faces foreground
    let pt = faces[i];
    let GC_PR = [
      new cv.Point(pt.x, pt.y - pt.height / 3),
      new cv.Point(pt.x + pt.width, pt.y + pt.height * 3)
    ];
    let GC = [
      new cv.Point(pt.x + pt.width / 3, pt.y),
      new cv.Point(pt.x + pt.width - pt.width / 3, pt.y + pt.height * 3) //*3 head-heights in assumed body
    ];

    cv.rectangle(
      mask,
      GC_PR[0],
      GC_PR[1],
      new cv.Scalar(cv.GC_PR_FGD),
      -1,
      4,
      0
    );
    cv.rectangle(mask, GC[0], GC[1], new cv.Scalar(cv.GC_FGD), -1, 4, 0);
  }

  let bgdModel = new cv.Mat();
  let fgdModel = new cv.Mat();
  let rect = new cv.Rect(D, D, sto.width, sto.height);

  let sampleSize = false;
  try {
    cv.grabCut(src, mask, rect, bgdModel, fgdModel, 2, cv.GC_INIT_WITH_MASK);
    sampleSize = true;
  } catch (err) {
    console.warn("no obvious fg/bg");
  }

  bgdModel.delete();
  fgdModel.delete();

  //draw grab rect
  //let point1 = new cv.Point(rect.x, rect.y);
  //let point2 = new cv.Point(rect.x + rect.width, rect.y + rect.height);
  //cv.rectangle(src, point1, point2, new cv.Scalar(0, 0, 255));

  //close holes, reduce artefacts (not responsive)
  cv.medianBlur(mask, mask, 3);

  let alphaMap = src.clone();
  var promise = new Promise(function (resolve) {
    if (sampleSize) {
      //draw alpha
      cv.cvtColor(src, src, cv.COLOR_RGB2RGBA);
      for (let i = 0; i < src.rows; i++) {
        for (let j = 0; j < src.cols; j++) {
          if (
            (mask.ucharPtr(i, j)[0] === 0 || mask.ucharPtr(i, j)[0] === 2) &&
            touchup.ucharPtr(i, j)[1] < 128
            //GC.GreenScreen(i, j)
          ) {
            //threejs.org/docs/#api/en/materials/MeshDistanceMaterial.alphaMap
            alphaMap.ucharPtr(i, j)[0] = 0;
            alphaMap.ucharPtr(i, j)[1] = 0;
            alphaMap.ucharPtr(i, j)[2] = 0;
            src.ucharPtr(i, j)[3] = 0;
          } else {
            alphaMap.ucharPtr(i, j)[0] = 255;
            alphaMap.ucharPtr(i, j)[1] = 255;
            alphaMap.ucharPtr(i, j)[2] = 255;
          }
        }
      }
    }
    touchup.delete();

    cv.imshow("alpha", alphaMap);
    cv.imshow("chroma", src);

    src.delete();
    mask.delete();
    alphaMap.delete();

    resolve("GrabCut => Segmentation");
  });

  promise.then(function (value) {
    // expected output: "GrabCut => Segmentation"
    console.log(value);
    //sto.dst = document.getElementById("chroma").toDataURL("image/png");
    //alpha = document.getElementById("alpha").toDataURL("image/png");
    segmentation();
  });
}

function segmentation() {
  console.log("segmentation");
  //todo: predominantly black, transparent, or custom kmeans/histogram

  let src = cv.imread("chroma");

  cv.cvtColor(src, src, cv.COLOR_RGBA2RGB, 0); //re-comment for alpha output
  let dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC3);

  cv.cvtColor(src, dst, cv.COLOR_RGB2GRAY, 0);
  cv.threshold(dst, dst, 0, 255, cv.THRESH_BINARY);

  //1st pad boundaries
  let kOdd = 2 * Math.floor(sto.width * 0.005) + 1;
  cv.GaussianBlur(dst, dst, new cv.Size(kOdd, kOdd), 0, 0, cv.BORDER_DEFAULT);
  //2nd reduce noise
  let anchor = new cv.Point(-1, -1);
  let M = cv.Mat.ones(5, 5, cv.CV_8U);
  cv.morphologyEx(
    dst,
    dst,
    cv.MORPH_OPEN,
    M,
    anchor,
    1,
    cv.BORDER_CONSTANT,
    cv.morphologyDefaultBorderValue()
  );
  M.delete();

  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(
    dst,
    contours,
    hierarchy,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
  );

  let hull = new cv.Mat();
  let defect = new cv.Mat();

  let colorBleed = new cv.Scalar(0, 255, 255);
  let colorPoint = new cv.Scalar(0, 255, 0);

  cv.drawContours(dst, contours, -1, colorBleed, cv.FILLED);
  dst.delete();

  var promise = new Promise(function (resolve) {
    for (let i = 0; i < contours.size(); ++i) {
      let cnt = contours.get(i);

      cv.convexHull(cnt, hull, false, false);
      cv.convexityDefects(cnt, hull, defect);

      sto.contours[i] = {
        SHAPE: [],
        BONES: [],
        CLUSTER: []
      };

      //area to test shallow cavity and skin weight
      //todo: minimum area threshold (noise)
      let area = Math.sqrt(cv.contourArea(cnt));
      //contour center
      let M = cv.moments(cnt),
        cX = Math.round(M.m10 / M.m00),
        cY = Math.round(M.m01 / M.m00);
      let center = new cv.Point(cX, cY);

      cv.circle(src, center, 6, colorPoint, -1);

      sto.contours[i].BONES.push([cX, cY, "root", Math.round(area)]);

      for (let j = 0; j < defect.rows; ++j) {
        let start = new cv.Point(
          cnt.data32S[defect.data32S[j * 4] * 2],
          cnt.data32S[defect.data32S[j * 4] * 2 + 1]
        );
        let end = new cv.Point(
          cnt.data32S[defect.data32S[j * 4 + 1] * 2],
          cnt.data32S[defect.data32S[j * 4 + 1] * 2 + 1]
        );
        let far = new cv.Point(
          cnt.data32S[defect.data32S[j * 4 + 2] * 2],
          cnt.data32S[defect.data32S[j * 4 + 2] * 2 + 1]
        );

        cv.line(src, start, end, colorBleed, 1, cv.LINE_4, 0); //convex hull

        let ptTest = Math.abs(
          cv.pointPolygonTest(
            cnt,
            new cv.Point((start.x + end.x) / 2, (start.y + end.y) / 2),
            true
          )
        );

        //todo: center end bone, align midbone perpendicular
        if (
          area / ((sto.width + sto.height) / 2) < 0.05 ||
          ptTest / area < 0.02
        ) {
          console.log("not minimum area");
        } else {
          sto.contours[i].CLUSTER.push({
            start: {
              x: start.x,
              y: start.y
            },
            end: {
              x: end.x,
              y: end.y
            },
            far: {
              x: far.x,
              y: far.y
            }
          });

          //cv.circle(src, start, 12, colorBleed, 1);
          cv.circle(src, end, 6, colorBleed, 1); //convex
          cv.circle(src, far, 6, colorBleed, 1); //concave
          cv.circle(
            src,
            new cv.Point((start.x + end.x) / 2, (start.y + end.y) / 2),
            4,
            colorBleed,
            1
          ); //convex midpoint (of cluster clockwise, last)
        }

        //console.log(end.x,'y',end.y);
      }
    }

    var obj = sto.contours;

    for (let i = 0; i < obj.length; i++) {
      let cl = obj[i].CLUSTER;
      let bones = [];
      for (let j = 0; j < cl.length; j++) {
        let pt = cl[j],
          nx = cl[j + 1] || cl[0];

        let avg = {
          x: Math.round(pt.far.x + nx.far.x) / 2,
          y: Math.round(pt.far.y + nx.far.y) / 2
        };

        bones[j] = [];
        //concave midpoint
        cv.circle(src, new cv.Point(avg.x, avg.y), 4, colorPoint, 1);
        bones[j].push([avg.x, avg.y, [avg.x, avg.y]]);

        //bones use relative x,y offset from previous
        let midbone = [(avg.x + pt.end.x) / 2, (avg.y + pt.end.y) / 2];
        cv.circle(src, new cv.Point(midbone[0], midbone[1]), 4, colorPoint, -1);

        let prev = bones[j][bones[j].length - 1];
        bones[j].push([
          midbone[0] - prev[0],
          midbone[1] - prev[1],
          [midbone[0], midbone[1]]
        ]);

        //convex
        bones[j].push([
          pt.end.x - prev[0],
          pt.end.y - prev[1],
          [pt.end.x, pt.end.y]
        ]);
      }

      sto.contours[i].BONES.push(bones);
    }

    let poly = new cv.MatVector();
    // approximates each contour to polygon
    for (let i = 0; i < contours.size(); ++i) {
      let tmp = new cv.Mat();
      let cnt = contours.get(i);
      //simplify path
      //docs.opencv.org/trunk/js_contour_features_approxPolyDP.html
      cv.approxPolyDP(cnt, tmp, 2, true);
      poly.push_back(tmp);

      var shape = poly.get(i).data32S;
      if (shape.length >= 8) {
        sto.contours[i].SHAPE = shape;
      }

      cnt.delete();
      tmp.delete();
    }
    //console.info('sto.contours', sto.contours);

    let alphaMap = cv.imread("alpha");
    const D = (sto.width + sto.height) / 128;
    // draw contours with random Scalar
    for (let i = 0; i < contours.size(); ++i) {
      let color = new cv.Scalar(
        255,
        Math.round(Math.random() * 128),
        Math.round(Math.random() * 128)
      );
      cv.drawContours(src, poly, i, color, 1, 8, hierarchy, 0);
      //draw contour edges on alpha map to close shadow
      cv.drawContours(
        alphaMap,
        poly,
        i,
        [255, 255, 255, 255],
        D,
        8,
        hierarchy,
        0
      );
    }
    hierarchy.delete();
    contours.delete();

    //sto.dst = document.getElementById('chroma').toDataURL('image/png'); //TEST registration
    cv.imshow("chroma", src);
    cv.imshow("alpha", alphaMap);

    src.delete();
    alphaMap.delete();

    hull.delete();
    defect.delete();

    resolve("Segmentation => Three.js");
  });

  promise.then(function (value) {
    // expected output: "Segmentation => Three.js"
    console.log(value);

    three();
  });
}

function getColor(canvas, x, y) {
  let mat = cv.imread(canvas);
  let colour = mat.ucharPtr(y, x);
  colour = new THREE.Color(
    "rgb(" + colour[0] + ", " + colour[1] + ", " + colour[2] + ")"
  );
  mat.delete();
  return colour;
}

//==========//==========//==========//==========//
//THREE.JS######################################//
//==========//==========//==========//==========//
function three() {
  if (!scene) {
    initScene();
    camera.position.set(0.5, -0.5, 2.0);
    orbit.update();
    render();
  }

  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.setSize(sto.width, sto.height);
  camera.aspect = sto.width / sto.height;

  group = new THREE.Group();
  group.name = "meshes";
  scene.add(group);
  group2 = new THREE.Group();
  group2.name = "artefact";
  group3 = new THREE.Group();
  group3.name = "raytest";
  group3.visible = false;
  sprite = new THREE.Group();
  sprite.name = "sprite";
  sprite.renderOrder = 100;
  sprite.visible = false;
  group.add(group2, group3, sprite);

  var textureLoader = new THREE.TextureLoader();
  textureLoader.crossOrigin = true;
  textureLoader.load(
    document.getElementById("chromaImg").src,
    function (texture) {
      texture.anisotropy = 8;

      let alphaMap = new THREE.CanvasTexture(document.getElementById("alpha"));
      alphaMap.offset.set(0, 1);

      material = new THREE.MeshStandardMaterial({
        color: 0x808080,
        wireframe: false,
        skinning: true,
        side: THREE.FrontSide, //hide gaps in skinmesh
        map: texture,
        transparent: true,
        //premultipliedAlpha: true, //texture alpha transparency
        alphaMap: alphaMap,
        alphaTest: 0.5,
        //extra
        dithering: true,
        bumpMap: texture,
        roughnessMap: texture,
        metalnessMap: texture,
        bumpScale: 0.01,
        roughness: 0.8,
        metalness: 0.5
      });
      texture.offset.set(0, 1);

      //loop contours for shape/bones/mesh
      let contours = sto.contours;
      for (let i = 0; i < sto.contours.length; i++) {
        //www.adrianboeing.com/demoscene/test/particleimage/canvas_particles_image.html
        let color = getColor(
          "chromaImg",
          contours[i].BONES[0][0],
          contours[i].BONES[0][1]
        );

        materialSides = new THREE.MeshStandardMaterial({
          color: color,
          skinning: true
        });

        initBones(contours[i]); //this is the engine
      }

      complete(timerGui);

      function complete(timer) {
        clearTimeout(timer);
        if (
          group.children.length + group2.children.length >=
          sto.contours.length * 2 + 1
        ) {
          console.log(sto, scene.children);
          //UI enabled
          document.documentElement.removeAttribute("class");
          document.getElementsByTagName("fieldset")[0].disabled = false;
          sto.update(false);
          setupDatGui();

          return;
        }
        timer = setTimeout(complete, 500);
      }
    }
  );

  function initScene() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(30, sto.width / sto.height, 0.01, 100);

    renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setSize(sto.width, sto.height);
    renderer.setClearColor("#00FF00", 0.0);

    document.getElementById("rig").appendChild(renderer.domElement);

    orbit = new OrbitControls(camera, renderer.domElement);
    orbit.target.set(0.5, -0.5, 0);
    //lights
    renderer.physicallyCorrectLights = true;
    var light = new THREE.AmbientLight(0xffffff, 2);
    var lightShadow = new THREE.PointLight(0xffffff, 5, 4, 2);
    lightShadow.position.set(0.5, 0.5, 0.5);
    lightShadow.castShadow = true;
    lightShadow.shadow.mapSize.set(1024, 1024);

    var lightShadowHelper = new THREE.PointLightHelper(lightShadow, 0.25);
    scene.add(light, lightShadow, lightShadowHelper);

    //plane
    var planeGeometry = new THREE.PlaneBufferGeometry(2, 2);
    planeGeometry.rotateX(Math.PI / 2);
    var planeMaterial = new THREE.ShadowMaterial({
      opacity: 1
    });
    var plane = new THREE.Mesh(planeGeometry, planeMaterial);
    plane.position.set(0.5, -1, 0);
    plane.rotation.z = Math.PI;
    plane.receiveShadow = true;
    plane.name = "floor";
    //grid
    var helper = new THREE.GridHelper(2, 2);
    helper.position.set(0.5, -1, 0);
    helper.material.opacity = 0.25;
    helper.material.transparent = true;
    helper.receiveShadow = true;
    scene.add(plane, helper);

    scene.fog = new THREE.FogExp2(0xf0f0f0, 0.1);
  }

  let resizeTimer;
  window.addEventListener(
    "resize",
    function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        sto.update(false);
      }, 250);
    },
    false
  );

  function initBones(contour) {
    contour.BONES[0][3];
    let skinArr = createGeometry(contour.SHAPE, contour.BONES[0][3]);
    let geometry = skinArr[0],
      shadow = skinArr[1];
    let bones = createBones(contour.BONES);

    createBoundBox(bones, contour);
    createMesh(bones, geometry, shadow);
    sto.boneIndexes = {};
  }

  function createBoundBox(bones, contour) {
    //cavity zone to raytest vertex to bone
    let defects = contour.CLUSTER;
    let cavity = [];
    for (let i = 0; i < defects.length; i++) {
      cavity[i] = {};

      cavity[i].convex = {
        x: (defects[i].start.x + defects[i].end.x) / 2 / sto.width,
        y: (-defects[i].start.y + -defects[i].end.y) / 2 / sto.height
      };
      cavity[i].concave = {
        x: defects[i].far.x / sto.width,
        y: -defects[i].far.y / sto.height
      };
      cavity[i].end = {
        x: defects[i].end.x / sto.width,
        y: -defects[i].end.y / sto.height
      };
      cavity[i].midpoint = {
        x: (cavity[i].convex.x + cavity[i].concave.x) / 2,
        y: (cavity[i].convex.y + cavity[i].concave.y) / 2
      };
    }

    let root = new THREE.Shape();
    for (let i = 0; i < defects.length; i++) {
      //bounding boxes for raytest skinmesh
      let bone0 = new THREE.Shape();
      let bone1 = new THREE.Shape();

      let src = cavity[i];

      let dst = i == defects.length - 1 ? cavity[0] : cavity[i + 1];
      //console.log(src, dst);

      //bone root (inner hull)
      if (i == 0) {
        root.moveTo(src.concave.x, src.concave.y);
      }
      root.lineTo(dst.concave.x, dst.concave.y);
      //Bone_0
      bindVertex = bones[i * 3 + 0];
      bindBone = bones[i * 3 + 1];
      bone0.moveTo(src.concave.x, src.concave.y);
      bone0.lineTo(src.midpoint.x, src.midpoint.y);
      bone0.lineTo(dst.midpoint.x, dst.midpoint.y);
      bone0.lineTo(dst.concave.x, dst.concave.y);
      bone0.closePath();
      extrude(bone0, bindVertex, bindBone);
      //Bone_1
      bindVertex = bones[i * 3 + 1];
      bindBone = bones[i * 3 + 2];
      bone1.moveTo(src.convex.x, src.convex.y);
      bone1.lineTo(src.midpoint.x, src.midpoint.y);
      bone1.lineTo(dst.midpoint.x, dst.midpoint.y);
      bone1.lineTo(dst.convex.x, dst.convex.y);
      //googleprojectzero.blogspot.com/2019/02/the-curious-case-of-convexity-confusion.html
      let slope = {
        x: src.end.x - bones[0].position.x,
        y: src.end.y - bones[0].position.y
      };

      bone1.lineTo(src.end.x + slope.x, src.end.y + slope.y);
      bone1.closePath();
      extrude(bone1, bindVertex, bindBone);
    }
    bindVertex = bones[0];
    bindBone = bones[0];

    function extrude(shape, bindVertex, bindBone) {
      let geom = new THREE.ExtrudeGeometry(shape, {
        steps: 1,
        depth: 0.1,
        bevelEnabled: false
      });
      geom = new THREE.BufferGeometry().fromGeometry(geom);
      let hit = new THREE.Mesh(
        geom,
        new THREE.MeshLambertMaterial({
          side: THREE.BackSide, //precisely
          opacity: 0.125,
          color: 0x00ffff,
          transparent: true
        })
      );
      //this is 1-off...
      hit.bindVertex = bindVertex.id;
      hit.bindBone = bindBone.id;

      group3.add(hit);

      hit.position.z = -0.05;
    }
  }

  function createGeometry(shape, area) {
    var segment = new THREE.Shape();
    segment.autoClose = true;
    for (let j = 0; j < shape.length; j += 2) {
      if (j < 2) {
        segment.moveTo(shape[j] / sto.width, -shape[j + 1] / sto.height);
      } else {
        segment.lineTo(shape[j] / sto.width, -shape[j + 1] / sto.height);
      }
    }

    //shadow alpha helper
    shadow = new THREE.ShapeGeometry(segment);
    shadow = new THREE.BufferGeometry().fromGeometry(shadow);

    //opt1: extrude
    geometry = new THREE.ExtrudeGeometry(segment, {
      steps: 1,
      depth: 0.05,
      bevelEnabled: false
    });

    //opt2: tessellate/subdivide
    //note: balance performance versus quality
    let maxEdgeLength = area / sto.width / 8;
    var tessellateModifier = new TessellateModifier(maxEdgeLength);
    for (let k = 0; k < 8; k++) {
      tessellateModifier.modify(geometry);
    }

    geometry = new THREE.BufferGeometry().fromGeometry(geometry);

    return [geometry, shadow];
  }

  function createBones(sizing) {
    let bones = [];

    var prevBone = new THREE.Bone();
    prevBone.cardinal = "axis";
    prevBone.name = "Root";
    bones.push(prevBone);
    sto.boneIndexes[prevBone.id] = bones.length;
    sto.bones.push(prevBone);

    prevBone.position.x = sizing[0][0] / sto.width;
    prevBone.position.y = -sizing[0][1] / sto.height;
    prevBone.positionGlobal = {
      x: sizing[0][0] / sto.width,
      y: -sizing[0][1] / sto.height
    };
    prevBone.dist = [];

    let ext = sizing[1];
    //extremity
    for (let i = 0; i < ext.length; i++) {
      let jnt = ext[i];
      //joint
      for (let j = 0; j < jnt.length; j++) {
        origin = prevBone.name == "Bone_0" ? 0 : 1; // origin offset
        var bone = new THREE.Bone();
        bone.position.x = jnt[j][0] / sto.width - origin * prevBone.position.x;
        bone.position.y =
          -jnt[j][1] / sto.height - origin * prevBone.position.y;
        bone.positionGlobal = {
          x: jnt[j][2][0],
          y: jnt[j][2][1]
        };
        bone.dist = [];
        bone.position.z = 0;
        bone.cardinal = bone.position.y > 0 ? "N" : "S";
        if (Math.abs(bone.position.x) > 0.02) {
          bone.cardinal += bone.position.x < 0 ? "W" : "E";
        }
        bone.name = "Bone_" + j;
        bones.push(bone);
        sto.boneIndexes[bone.id] = bones.length;
        sto.bones.push(bone);
        prevBone.add(bone);
        prevBone = bone;
      }
      prevBone = bones[0]; //next extremity reference parent
    }

    return bones;
  }

  function createMesh(bones, geometry, shadow) {
    //isolate artefacts such as convex shapes and noise
    let groupIn = bones[0].children.length !== 0 ? group : group2;

    let mesh = new THREE.SkinnedMesh(geometry, [material, materialSides]);
    mesh.position.z = -0.025;

    let meshLOD = new THREE.SkinnedMesh(
      shadow,
      new THREE.MeshStandardMaterial({
        transparent: true,
        opacity: 0,
        alphaMap: material.alphaMap,
        alphaTest: material.alphaTest,
        side: THREE.DoubleSide,
        skinning: true
      })
    );
    meshLOD.name = "shadow";

    //distance material for texture alpha
    //2d helper avoids 3d shadow/map quirks
    //threejs.org/examples/#webgl_shadowmap_pointlight
    var distanceMaterial = new THREE.MeshDistanceMaterial({
      alphaMap: material.alphaMap,
      alphaTest: material.alphaTest,
      skinning: material.skinning
    });
    meshLOD.customDistanceMaterial = distanceMaterial;

    let skeleton = new THREE.Skeleton(bones);

    var rootBone = skeleton.bones[0];

    meshLOD.add(rootBone);
    meshLOD.bind(skeleton);

    mesh.add(rootBone);
    mesh.bind(skeleton);

    let skeletonHelper = new THREE.SkeletonHelper(mesh);
    skeletonHelper.material.linewidth = 2;
    //note: skeletonHelper shows only for last bound mesh
    groupIn.add(mesh, meshLOD, skeletonHelper);

    mesh.receiveShadow = true;
    meshLOD.castShadow = true;

    rootBone.bind = {
      id: "",
      original: mesh.id
    };

    //attach raytest mesh to bone
    for (let h = 0; h < group3.children.length; h++) {
      bound = group3.children[h];
      if (bound.bindVertex) {
        let bone = scene.getObjectById(bound.bindBone);

        let pos = new THREE.Vector3();
        bound.getWorldPosition(pos);
        bound.parent = bone; //now parent is SkinnedMesh

        let posUp = new THREE.Vector3();
        bound.getWorldPosition(posUp);

        //invert translate
        bound.position.set(
          bound.position.x + (pos.x - posUp.x),
          bound.position.y + (pos.y - posUp.y),
          bound.position.z + (pos.z - posUp.z)
        );

        bound.updateMatrixWorld();
      }
    }

    function collision(from, to) {
      // calculate objects intersecting the picking ray
      var raycaster = new THREE.Raycaster();
      raycaster.set(from, to);
      var intersects = raycaster.intersectObjects(group3.children);
      //console.log(intersects);
      if (intersects.length > 0) {
        return intersects[0];
      }
      return false;
    }

    skinify(geometry);
    skinify(shadow);

    function skinify(geometry) {
      let dist = [];
      var skinIndices = [];
      var skinWeights = [];
      let position = geometry.attributes.position;

      var vertex = new THREE.Vector3();
      console.info("vertices=" + position.count);

      for (let i = 0; i < position.count; i++) {
        vertex.fromBufferAttribute(position, i);

        if (groupIn == group2) {
          //console.info('artefact');
          skinIndices.push(0, 0, 0, 0);
          skinWeights.push(1, 0, 0, 0);
          continue;
        }

        //ray test vertex to bone, collision group3
        dist[i] = [];
        for (let j = 0; j < bones.length; j++) {
          if (bones[j].children.length > 0) {
            //not end-bone
            let defect = collision(vertex, bones[j].position);
            if (defect && defect.object.bindVertex) {
              //todo: defect.distance>1e-17 && defect.distance<1
              //console.log(defect);
              dist[i].push(defect.distance + "|" + defect.object.bindVertex);
            }
          }
        }
        dist[i].sort();
        //console.log(dist[i]);

        //combined distances normalized to 1
        let norm = 0;
        for (let k = 0; k < 4; k++) {
          let close = dist[i];
          close =
            close[k] != undefined ? close[k].split("|") : [0.999, bones[0].id]; //0.999 prevents NaN
          dist[i][k] = {
            sWeight: 1 - close[0],
            sIndice: close[1] * 1
          };

          norm += dist[i][k].sWeight;
        }
        norm = 1 / norm;

        //stackoverflow.com/questions/23052306/what-is-the-meaning-of-skin-indices-and-skin-weights
        for (let l = 0; l < 4; l++) {
          skinIndices.push(sto.boneIndexes[dist[i][l].sIndice]);
          skinWeights.push(dist[i][l].sWeight * norm);
        }
      }

      //console.log(skinIndices, skinWeights);
      geometry.setAttribute(
        "skinIndex",
        new THREE.Uint16BufferAttribute(skinIndices, 4)
      );
      geometry.setAttribute(
        "skinWeight",
        new THREE.Float32BufferAttribute(skinWeights, 4)
      );
    }
  }

  function setupDatGui() {
    // todo: add multiple files to scene
    // mask/replace current until new file
    // animate from scene bones
    console.log("GUI");
    //workshop.chromeexperiments.com/examples/gui/
    if (gui) gui.destroy();
    gui = new GUI();
    gui.close();

    var folder = gui.addFolder("General Options");
    folder.open();
    var artefact = gui.addFolder("Artefacts");
    var root;

    folder.add(sto.state, "animateBones").name("animate");
    folder.add(sprite, "visible").name("bone.id");
    folder.add(material, "wireframe");
    folder.add(group3, "visible").name("raytest");

    for (let i = 0; i < sto.bones.length; i++) {
      let bone = sto.bones[i];

      makeTextSprite(bone);

      if (bone.name === "Root") {
        folder = bone.children.length > 0 ? gui : artefact;
      } else if (bone.name === "Bone_0") {
        folder = root;
      }

      folder = folder.addFolder(
        bone.name + "_" + bone.id + "__" + bone.cardinal
      );

      if (bone.name === "Root") {
        root = folder;

        folder
          .add(bone.parent, "pose")
          .onChange(bone.parent.pose())
          .onChange(function () {
            for (let j = 0; j < sprite.children.length; j++) {
              sprite.children[j].scale.set(1 / 30, 1 / 30, 1 / 30);
            }
          });
        folder
          .add(bone.bind, "id")
          .onFinishChange(function (value) {
            let bind = scene.getObjectById(Number(value));
            scene.getObjectById(Number(bone.id)).parent = bind
              ? bind
              : scene.getObjectById(Number(bone.bind.original));
          })
          .name("parent.id");

        folder
          .add(bone.position, "x", -2 + bone.position.x, 2 + bone.position.x)
          .name("position.x");
        folder
          .add(bone.position, "y", -2 + bone.position.y, 2 + bone.position.y)
          .name("position.y");
        folder
          .add(bone.position, "z", -2 + bone.position.z, 2 + bone.position.z)
          .name("position.z");

        folder.add(bone.scale, "x", 0, 2).name("scale.x");
        folder.add(bone.scale, "y", 0, 2).name("scale.y");
        folder.add(bone.scale, "z", 0, 2).name("scale.z");
      }

      let rot = Math.PI;
      folder.add(bone.rotation, "x", -rot, rot).name("rotation.x");
      folder.add(bone.rotation, "y", -rot, rot).name("rotation.y");
      folder.add(bone.rotation, "z", -rot, rot).name("rotation.z");
    }
  }

  var last = 0;
  var cycle = 1;

  function render(timestamp) {
    requestAnimationFrame(render);

    let amount = 0.001;
    //console.log(timestamp);
    if (timestamp - last > 2000) {
      last = timestamp;
      cycle *= -1;
    }

    //Wiggle the bones
    if (sto.state.animateBones) {
      for (let i = 0; i < sto.bones.length; i++) {
        let bone = sto.bones[i];
        let bi = bone.cardinal.indexOf("E") >= 0 ? 1 : -1;

        if (bone.cardinal == "axis") {
          //bone.rotation.y += (amount * cycle);
        } else if (bone.name == "Bone_0") {
          if (bone.cardinal == "N") {
            bone.rotation.x += 4 * (amount * cycle);
          } else if (bone.cardinal == "SW" || bone.cardinal == "SE") {
            bone.rotation.x += 4 * bi * (amount * cycle);
            let child = bone.children[0];
            child.rotation.x += 4 * bi * (amount * cycle);
          } else if (bone.cardinal == "NW" || bone.cardinal == "NE") {
            bone.parent.rotation.x *= bi;
            bone.children[0].rotation.x += 2 * bi * (amount * -cycle);
            let child = bone.children[0];
            child.rotation.x += 2 * bi * (amount * -cycle);
          }
        }
      }
    }
    renderer.render(scene, camera);
  }

  function makeTextSprite(bone) {
    let canvas = document.createElement("canvas");
    let ctx = canvas.getContext("2d");

    // text
    canvas.width = 64;
    canvas.height = canvas.width * 0.5;
    ctx.fillStyle = "cyan";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = canvas.height + "px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#000000";
    ctx.fillText(bone.id, canvas.width / 2, canvas.height / 2);

    // canvas contents will be used for a texture
    var texture = new THREE.Texture(canvas);
    texture.needsUpdate = true;

    var spriteMaterial = new THREE.SpriteMaterial({
      map: texture,
      sizeAttenuation: false,
      depthTest: false
    });
    var boneId = new THREE.Sprite(spriteMaterial);

    boneId.rotateX(Math.PI / 2);

    boneId.position.set(
      bone.position.x / sto.width,
      bone.position.y / sto.height,
      0
    );
    boneId.scale.divideScalar(30);
    sprite.add(boneId);

    boneId.parent = scene.getObjectById(bone.id);
    canvas = null;
  }
}

var touchupButton = document.getElementById("touchupMask");
touchupButton.addEventListener("click", grabCut);
//www.createjs.com/demos/easeljs/curveto
var touchup, stage;
var drawingCanvas;
var oldPt;
var oldMidPt;
var color;
var stroke;
var index;

function init() {
  touchup = document.getElementById("touchup");
  index = 0;

  //check to see if we are running in a browser with touch support
  stage = new createjs.Stage(touchup);
  stage.autoClear = false;
  stage.enableDOMEvents(true);

  createjs.Touch.enable(stage);
  createjs.Ticker.framerate = 15;

  drawingCanvas = new createjs.Shape();

  stage.addEventListener("stagemousedown", handleMouseDown);
  stage.addEventListener("stagemouseup", handleMouseUp);

  stage.addChild(drawingCanvas);
  stage.update();
}

function handleMouseDown() {
  color = document.getElementById("color").value;
  stroke = 5;
  oldPt = new createjs.Point(stage.mouseX, stage.mouseY);
  oldMidPt = oldPt;
  stage.addEventListener("stagemousemove", handleMouseMove);
}

function handleMouseMove() {
  var midPt = new createjs.Point(
    (oldPt.x + stage.mouseX) >> 1,
    (oldPt.y + stage.mouseY) >> 1
  );

  drawingCanvas.graphics
    .clear()
    .setStrokeStyle(stroke, "round", "round")
    .beginStroke(color)
    .moveTo(midPt.x, midPt.y)
    .curveTo(oldPt.x, oldPt.y, oldMidPt.x, oldMidPt.y);

  oldPt.x = stage.mouseX;
  oldPt.y = stage.mouseY;

  oldMidPt.x = midPt.x;
  oldMidPt.y = midPt.y;

  stage.update();
}

function handleMouseUp() {
  stage.removeEventListener("stagemousemove", handleMouseMove);
}

init();

var exporter = new GLTFExporter();

function export3d() {
  let name = document.getElementById("photo").files[0];
  name = name ? name.name : document.getElementById("chromaImg").src;
  name = name.slice(0, name.lastIndexOf(".")).slice(name.lastIndexOf("/") + 1);

  // Parse the input and generate the glTF output
  for (let i = 0; i < sto.bones.length; i++) {
    if (sto.bones[i].name != "Root") {
      //todo: if no children, check min area?
      continue;
    }

    let limbs = sto.bones[i].children.length;

    exporter.parse(
      sto.bones[i].parent,
      function (gltf) {
        console.log(name, gltf);
        saveString(JSON.stringify(gltf), name + "_" + limbs + ".glb");
      },
      {
        forceIndices: true
      }
    );
  }
}

function save(blob, filename) {
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
}

function saveString(text, filename) {
  save(
    new Blob([text], {
      type: "text/plain"
    }),
    filename
  );
}

var link = document.createElement("a");
link.style.display = "none";
document.body.appendChild(link);

var exportButton = document.getElementById("export3d");
exportButton.addEventListener("click", export3d);
