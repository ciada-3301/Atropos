/* ============================================================
   Minimal Orbit Controls for Three.js r128
   ------------------------------------------------------------
   A small, dependency-free orbit/pan/zoom controller, written
   directly against the Three.js r128 API rather than pulled
   from a CDN mirror (the official examples/jsm build isn't
   reliably available as a single classic <script> include for
   r128 across CDNs, so this avoids a fragile external dependency).

   Supports:
     - left-drag: orbit
     - right-drag: pan
     - wheel: zoom (dolly)
     - touch: one-finger orbit, two-finger pinch/pan
   ============================================================ */

function SimpleOrbitControls(camera, domElement) {
  const scope = this;

  this.target = new THREE.Vector3(0, 0, 0);
  this.minDistance = 0.5;
  this.maxDistance = 20;
  this.rotateSpeed = 1.0;
  this.zoomSpeed = 1.0;
  this.panSpeed = 1.0;
  this.enableDamping = true;
  this.dampingFactor = 0.08;

  let spherical = new THREE.Spherical();
  let sphericalDelta = new THREE.Spherical(0, 0, 0);
  let panOffset = new THREE.Vector3();
  let scale = 1;

  const offset = new THREE.Vector3();
  const quat = new THREE.Quaternion().setFromUnitVectors(camera.up, new THREE.Vector3(0, 1, 0));
  const quatInverse = quat.clone().invert();

  let state = "none";
  let rotateStart = new THREE.Vector2();
  let rotateEnd = new THREE.Vector2();
  let rotateDelta = new THREE.Vector2();
  let panStart = new THREE.Vector2();
  let panEnd = new THREE.Vector2();
  let panDelta = new THREE.Vector2();
  let dollyStart = new THREE.Vector2();
  let dollyEnd = new THREE.Vector2();

  function getZoomScale() {
    return Math.pow(0.95, scope.zoomSpeed);
  }

  function rotateLeft(angle) { sphericalDelta.theta -= angle; }
  function rotateUp(angle) { sphericalDelta.phi -= angle; }

  function panLeft(distance, m) {
    const v = new THREE.Vector3();
    v.setFromMatrixColumn(m, 0);
    v.multiplyScalar(-distance);
    panOffset.add(v);
  }

  function panUp(distance, m) {
    const v = new THREE.Vector3();
    v.setFromMatrixColumn(m, 1);
    v.multiplyScalar(distance);
    panOffset.add(v);
  }

  function pan(deltaX, deltaY) {
    const el = domElement;
    const v = new THREE.Vector3();
    v.copy(camera.position).sub(scope.target);
    let targetDistance = v.length();
    targetDistance *= Math.tan((camera.fov / 2) * Math.PI / 180.0);
    panLeft(2 * deltaX * targetDistance / el.clientHeight, camera.matrix);
    panUp(2 * deltaY * targetDistance / el.clientHeight, camera.matrix);
  }

  function dollyOut(dollyScale) { scale /= dollyScale; }
  function dollyIn(dollyScale) { scale *= dollyScale; }

  this.update = function () {
    const position = camera.position;
    offset.copy(position).sub(scope.target);
    offset.applyQuaternion(quat);
    spherical.setFromVector3(offset);

    spherical.theta += sphericalDelta.theta;
    spherical.phi += sphericalDelta.phi;
    spherical.phi = Math.max(0.001, Math.min(Math.PI - 0.001, spherical.phi));
    spherical.makeSafe();
    spherical.radius *= scale;
    spherical.radius = Math.max(scope.minDistance, Math.min(scope.maxDistance, spherical.radius));

    scope.target.add(panOffset);

    offset.setFromSpherical(spherical);
    offset.applyQuaternion(quatInverse);
    position.copy(scope.target).add(offset);
    camera.lookAt(scope.target);

    if (scope.enableDamping) {
      sphericalDelta.theta *= (1 - scope.dampingFactor);
      sphericalDelta.phi *= (1 - scope.dampingFactor);
      panOffset.multiplyScalar(1 - scope.dampingFactor);
    } else {
      sphericalDelta.set(0, 0, 0);
      panOffset.set(0, 0, 0);
    }
    scale = 1;
    return true;
  };

  function onMouseDown(event) {
    event.preventDefault();
    if (event.button === 0) {
      state = "rotate";
      rotateStart.set(event.clientX, event.clientY);
    } else if (event.button === 2) {
      state = "pan";
      panStart.set(event.clientX, event.clientY);
    }
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  }

  function onMouseMove(event) {
    event.preventDefault();
    if (state === "rotate") {
      rotateEnd.set(event.clientX, event.clientY);
      rotateDelta.subVectors(rotateEnd, rotateStart).multiplyScalar(scope.rotateSpeed);
      const el = domElement;
      rotateLeft(2 * Math.PI * rotateDelta.x / el.clientHeight);
      rotateUp(2 * Math.PI * rotateDelta.y / el.clientHeight);
      rotateStart.copy(rotateEnd);
    } else if (state === "pan") {
      panEnd.set(event.clientX, event.clientY);
      panDelta.subVectors(panEnd, panStart).multiplyScalar(scope.panSpeed);
      pan(panDelta.x, panDelta.y);
      panStart.copy(panEnd);
    }
  }

  function onMouseUp() {
    state = "none";
    document.removeEventListener("mousemove", onMouseMove);
    document.removeEventListener("mouseup", onMouseUp);
  }

  function onMouseWheel(event) {
    event.preventDefault();
    if (event.deltaY < 0) dollyIn(getZoomScale());
    else if (event.deltaY > 0) dollyOut(getZoomScale());
  }

  function onContextMenu(event) { event.preventDefault(); }

  let touchState = "none";
  let prevTouchDist = 0;

  function onTouchStart(event) {
    if (event.touches.length === 1) {
      touchState = "rotate";
      rotateStart.set(event.touches[0].clientX, event.touches[0].clientY);
    } else if (event.touches.length === 2) {
      touchState = "pinch";
      const dx = event.touches[0].clientX - event.touches[1].clientX;
      const dy = event.touches[0].clientY - event.touches[1].clientY;
      prevTouchDist = Math.sqrt(dx * dx + dy * dy);
    }
  }

  function onTouchMove(event) {
    event.preventDefault();
    if (touchState === "rotate" && event.touches.length === 1) {
      rotateEnd.set(event.touches[0].clientX, event.touches[0].clientY);
      rotateDelta.subVectors(rotateEnd, rotateStart).multiplyScalar(scope.rotateSpeed);
      const el = domElement;
      rotateLeft(2 * Math.PI * rotateDelta.x / el.clientHeight);
      rotateUp(2 * Math.PI * rotateDelta.y / el.clientHeight);
      rotateStart.copy(rotateEnd);
    } else if (touchState === "pinch" && event.touches.length === 2) {
      const dx = event.touches[0].clientX - event.touches[1].clientX;
      const dy = event.touches[0].clientY - event.touches[1].clientY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < prevTouchDist) dollyOut(getZoomScale());
      else if (dist > prevTouchDist) dollyIn(getZoomScale());
      prevTouchDist = dist;
    }
  }

  function onTouchEnd() { touchState = "none"; }

  domElement.addEventListener("mousedown", onMouseDown);
  domElement.addEventListener("wheel", onMouseWheel, { passive: false });
  domElement.addEventListener("contextmenu", onContextMenu);
  domElement.addEventListener("touchstart", onTouchStart, { passive: true });
  domElement.addEventListener("touchmove", onTouchMove, { passive: false });
  domElement.addEventListener("touchend", onTouchEnd);

  this.update();
}