/* ============================================================
   Atropos Explorer — 3D Viewport
   ------------------------------------------------------------
   Renders the reference embedding cloud, RGB axis gizmo, soft
   region labels, and the live query point. Exposes a small
   public API consumed by app.js.
   ============================================================ */

const AtroposViewport = (function () {
  let scene, camera, renderer, controls;
  let pointCloud, queryMesh, queryGlow;
  let regionSprites = [];
  let raycaster, mouse;
  let referencePoints = [];
  let phylumColors = [];
  let regionsVisible = true;
  let hoverCallback = null;
  let animFrame = null;
  let queryTrailLine = null;

  const CONTAINER_ID = "viewportCanvas";

  // A muted, instrument-appropriate categorical palette for taxa regions.
  // Deliberately avoids saturated primary hues (those are reserved for the
  // X/Y/Z axes) and avoids the "flashy purple/green/blue" look -- leans on
  // desaturated earth, clay, slate, and ochre tones instead.
  const PALETTE = [
    "#b5805a", "#7d9b8a", "#a3683f", "#6f8aa3", "#9b7a9e",
    "#8a9b5e", "#b56a6a", "#5e8a8f", "#a89456", "#6a7d9b",
    "#9b5e7a", "#7a9b6a", "#a37d5e", "#5e7a9b", "#9b8a5e",
    "#6a9b8a", "#a35e6a", "#7d8a9b", "#9b6a5e", "#5e9b7a",
  ];

  function colorForPhylumId(id) {
    return PALETTE[id % PALETTE.length];
  }

  function init(container) {
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x161616);
    scene.fog = new THREE.FogExp2(0x161616, 0.12);

    camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 100);
    camera.position.set(2.4, 1.8, 2.4);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    controls = new SimpleOrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.minDistance = 0.8;
    controls.maxDistance = 12;

    buildAxes();
    buildGridFloor();

    raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.035 };
    mouse = new THREE.Vector2(-10, -10);

    renderer.domElement.addEventListener("mousemove", onMouseMove);
    window.addEventListener("resize", () => onResize(container));

    animate();
  }

  function buildAxes() {
    const axisLength = 1.35;
    const group = new THREE.Group();

    const axes = [
      { dir: [1, 0, 0], color: 0xc1554a },
      { dir: [0, 1, 0], color: 0x4f9e63 },
      { dir: [0, 0, 1], color: 0x4a82b5 },
    ];

    axes.forEach(({ dir, color }) => {
      const points = [
        new THREE.Vector3(-dir[0] * axisLength, -dir[1] * axisLength, -dir[2] * axisLength),
        new THREE.Vector3(dir[0] * axisLength, dir[1] * axisLength, dir[2] * axisLength),
      ];
      const geom = new THREE.BufferGeometry().setFromPoints(points);
      const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.45 });
      group.add(new THREE.Line(geom, mat));

      // arrow tip
      const coneGeom = new THREE.ConeGeometry(0.025, 0.08, 10);
      const coneMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85 });
      const cone = new THREE.Mesh(coneGeom, coneMat);
      cone.position.set(dir[0] * axisLength, dir[1] * axisLength, dir[2] * axisLength);
      if (dir[0]) cone.rotation.z = -Math.PI / 2;
      if (dir[2]) cone.rotation.x = Math.PI / 2;
      group.add(cone);
    });

    scene.add(group);
  }

  function buildGridFloor() {
    const grid = new THREE.GridHelper(2.8, 14, 0x2a2a2a, 0x232323);
    grid.position.y = -1.3;
    scene.add(grid);
  }

  function buildPointCloud(points) {
    referencePoints = points;

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);

    const tmpColor = new THREE.Color();

    points.forEach((p, i) => {
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;

      tmpColor.set(colorForPhylumId(p.phylum_id));
      colors[i * 3] = tmpColor.r;
      colors[i * 3 + 1] = tmpColor.g;
      colors[i * 3 + 2] = tmpColor.b;
    });

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    // sizeAttenuation: false keeps each point a fixed size in SCREEN
    // PIXELS regardless of camera distance -- this is what makes a
    // scatter-plot-style point cloud feel right. With attenuation on
    // (the default), `size` is a WORLD-space size, so points visually
    // grow as the camera dollies closer during zoom, eventually filling
    // the screen like golf balls at close range. Pixel size is multiplied
    // by the renderer's pixel ratio so retina/high-DPI screens get a
    // consistent on-screen size too (renderer.setPixelRatio is already
    // set in init() -- this just keeps point size in sync with it).
    const material = new THREE.PointsMaterial({
      size: 4 * renderer.getPixelRatio(),
      vertexColors: true,
      transparent: true,
      opacity: 0.55,
      sizeAttenuation: false,
      depthWrite: false,
    });

    pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);
  }

  function buildRegionLabels(regions) {
    regionSprites.forEach((s) => scene.remove(s));
    regionSprites = [];

    regions.slice(0, 14).forEach((region) => {
      const sprite = makeTextSprite(region.phylum, colorForPhylumId(region.phylum_id));
      sprite.position.set(region.x, region.y + 0.06, region.z);
      scene.add(sprite);
      regionSprites.push(sprite);
    });
  }

  function makeTextSprite(text, colorHex) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const fontSize = 30;
    ctx.font = `500 ${fontSize}px Inter, sans-serif`;
    const textWidth = ctx.measureText(text).width;

    canvas.width = textWidth + 24;
    canvas.height = fontSize + 16;

    ctx.font = `500 ${fontSize}px Inter, sans-serif`;
    ctx.fillStyle = "rgba(22,22,22,0.72)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = colorHex;
    ctx.fillRect(0, 0, 3, canvas.height);
    ctx.fillStyle = "#e8e6e1";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 12, canvas.height / 2 + 1);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(material);
    const scale = 0.0026;
    sprite.scale.set(canvas.width * scale, canvas.height * scale, 1);
    sprite.renderOrder = 999;
    return sprite;
  }

  function setRegionsVisible(visible) {
    regionsVisible = visible;
    regionSprites.forEach((s) => (s.visible = visible));
  }

  function showQueryPoint(xyz) {
    clearQueryPoint();

    const geom = new THREE.SphereGeometry(0.032, 20, 20);
    const mat = new THREE.MeshBasicMaterial({ color: 0xc9a227 });
    queryMesh = new THREE.Mesh(geom, mat);
    queryMesh.position.set(xyz.x, xyz.y, xyz.z);
    scene.add(queryMesh);

    const glowGeom = new THREE.SphereGeometry(0.07, 20, 20);
    const glowMat = new THREE.MeshBasicMaterial({
      color: 0xc9a227, transparent: true, opacity: 0.22, depthWrite: false,
    });
    queryGlow = new THREE.Mesh(glowGeom, glowMat);
    queryGlow.position.copy(queryMesh.position);
    scene.add(queryGlow);

    // A faint vertical drop-line to the floor grid, like a map pin --
    // helps read the point's position relative to the cloud at a glance.
    const floorY = -1.3;
    const lineGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(xyz.x, xyz.y, xyz.z),
      new THREE.Vector3(xyz.x, floorY, xyz.z),
    ]);
    const lineMat = new THREE.LineDashedMaterial({
      color: 0xc9a227, transparent: true, opacity: 0.35, dashSize: 0.04, gapSize: 0.03,
    });
    queryTrailLine = new THREE.Line(lineGeom, lineMat);
    queryTrailLine.computeLineDistances();
    scene.add(queryTrailLine);

    animateCameraTo(xyz);
  }

  function clearQueryPoint() {
    if (queryMesh) { scene.remove(queryMesh); queryMesh = null; }
    if (queryGlow) { scene.remove(queryGlow); queryGlow = null; }
    if (queryTrailLine) { scene.remove(queryTrailLine); queryTrailLine = null; }
  }

  function animateCameraTo(xyz) {
    const startTarget = controls.target.clone();
    const endTarget = new THREE.Vector3(xyz.x, xyz.y, xyz.z);
    const duration = 700;
    const startTime = performance.now();

    function step(now) {
      const t = Math.min(1, (now - startTime) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      controls.target.lerpVectors(startTarget, endTarget, eased * 0.6);
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function onMouseMove(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  function checkHover() {
    if (!pointCloud || !hoverCallback) return;
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(pointCloud);
    if (intersects.length > 0) {
      const idx = intersects[0].index;
      hoverCallback(referencePoints[idx]);
    } else {
      hoverCallback(null);
    }
  }

  function onResize(container) {
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  }

  function animate() {
    animFrame = requestAnimationFrame(animate);
    controls.update();
    checkHover();

    if (queryGlow) {
      const pulse = 1 + Math.sin(performance.now() * 0.004) * 0.15;
      queryGlow.scale.set(pulse, pulse, pulse);
    }

    renderer.render(scene, camera);
  }

  function resetCamera() {
    const startPos = camera.position.clone();
    const endPos = new THREE.Vector3(2.4, 1.8, 2.4);
    const startTarget = controls.target.clone();
    const endTarget = new THREE.Vector3(0, 0, 0);
    const duration = 600;
    const startTime = performance.now();

    function step(now) {
      const t = Math.min(1, (now - startTime) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      camera.position.lerpVectors(startPos, endPos, eased);
      controls.target.lerpVectors(startTarget, endTarget, eased);
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  return {
    init,
    buildPointCloud,
    buildRegionLabels,
    setRegionsVisible,
    showQueryPoint,
    clearQueryPoint,
    resetCamera,
    colorForPhylumId,
    onHover: (cb) => { hoverCallback = cb; },
  };
})();