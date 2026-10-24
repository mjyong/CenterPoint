
import { CubeRefractionMapping } from "./lib/three.module.js";
import {saveWorldList} from "./save.js"

var Header=function(ui, data, cfg, onSceneChanged, onFrameChanged, onObjectSelected, onCameraChanged){

    this.ui = ui;
    this.data =  data;
    this.cfg = cfg;
    this.boxUi = ui.querySelector("#box");
    this.refObjUi = ui.querySelector("#ref-obj");
    this.sceneSelectorUi = ui.querySelector("#scene-selector");
    this.frameSelectorUi = ui.querySelector("#frame-selector");
    this.objectSelectorUi = ui.querySelector("#object-selector");
    this.cameraSelectorUi = ui.querySelector("#camera-selector");
    this.changedMarkUi = ui.querySelector("#changed-mark");

    this.onSceneChanged = onSceneChanged;
    this.onFrameChanged = onFrameChanged;
    this.onObjectSelected = onObjectSelected;
    this.onCameraChanged = onCameraChanged;


    if (cfg.disableSceneSelector){
        this.sceneSelectorUi.style.display="none";
    }

    if (cfg.disableFrameSelector){
        this.frameSelectorUi.style.display="none";
    }

    if (cfg.disableCameraSelector){
        this.cameraSelectorUi.style.display="none";
    }

    // update scene selector ui
    

    

    this.updateSceneList = function(sceneDescList){
        let scene_selector_str = "<option>--scene--</option>";
        // CenterPoint dog pipeline (2026-10-23): scenes exported with a split (desc.json "split", written by
        // tools/dog/sustechpoints_convert.py export --split) are grouped: validation first, then training,
        // then the rest. The order here is also the order "next frame" walks through scenes (editor.js).
        let groups = [["val", "验证集 val"], ["train", "训练集 train"], ["", "其它"]];
        let splitOf = (scene)=>{
            let desc = sceneDescList[scene];
            let split = desc && desc.split ? desc.split : "";
            return groups.some(g=>g[0] == split) ? split : "";
        };
        groups.forEach(([split, title])=>{
            let scenes = Object.keys(sceneDescList).filter(scene=>splitOf(scene) == split);
            if (scenes.length == 0)
                return;
            scene_selector_str += '<optgroup label="' + title + ' (' + scenes.length + ')">';
            scenes.forEach(scene=>{
                let desc = sceneDescList[scene];
                let text = scene + (desc && desc.scene ? " - " + desc.scene : "");
                scene_selector_str += "<option value="+scene +">"+text+ "</option>";
            });
            scene_selector_str += "</optgroup>";
        });

        this.ui.querySelector("#scene-selector").innerHTML = scene_selector_str;
    }

    // scene names in selector order (see updateSceneList)
    this.sceneOrder = function(){
        return Array.from(this.sceneSelectorUi.querySelectorAll("option"))
                    .map(o=>o.value).filter(v=>v && v != "--scene--");
    }
    
    this.updateSceneList(this.data.sceneDescList);

    this.ui.querySelector("#btn-reload-scene-list").onclick = (event)=>{
        let curentValue = this.sceneSelectorUi.value;
        
        this.data.readSceneList().then((sceneDescList=>{
            this.updateSceneList(sceneDescList);
            this.sceneSelectorUi.value = curentValue;
        }))
    }

    

    this.sceneSelectorUi.onchange = (e)=>{this.onSceneChanged(e);};
    this.objectSelectorUi.onchange = (e)=>{this.onObjectSelected(e);};
    this.frameSelectorUi.onchange = (e)=>{this.onFrameChanged(e);};
    this.cameraSelectorUi.onchange = (e)=>{this.onCameraChanged(e);};

    this.setObject = function(id)
    {
        this.objectSelectorUi.value = id;
    }

    this.clear_box_info = function(){
        this.boxUi.innerHTML = '';
    };
    
    this.update_box_info = function(box){
        var scale = box.scale;
        var pos = box.position;
        var rotation = box.rotation;
        var points_number = box.world.lidar.get_box_points_number(box);
        let distance = Math.sqrt(pos.x*pos.x + pos.y*pos.y).toFixed(2);

        this.boxUi.innerHTML = "<span>" + box.obj_type +"-"+box.obj_track_id + 
                               (box.annotator? ("</span> | <span title='annotator'>" + box.annotator) : "") +
                               "</span> | <span title='distance'>" + distance +
                               "</span> | <span title='position'>"+pos.x.toFixed(2) +" "+pos.y.toFixed(2) + " " + pos.z.toFixed(2) + 
                               "</span> | <span title='scale'>" +scale.x.toFixed(2) +" "+scale.y.toFixed(2) + " " + scale.z.toFixed(2) + 
                               "</span> | <span title='rotation'>" +
                                (rotation.x*180/Math.PI).toFixed(2)+" "+(rotation.y*180/Math.PI).toFixed(2)+" "+(rotation.z*180/Math.PI).toFixed(2)+
                                "</span> | <span title = 'points'>" +
                                points_number + "</span> ";
        // CenterPoint dog pipeline (2026-10-23): what the auto-label said about this box
        if (box.dog){
            let d = box.dog;
            // candidates (2026-10-24): the models that found the object and their scores instead
            this.boxUi.innerHTML += (d.candidate ?
                "| <span title='candidate: models that found it, with their scores'>候选 " +
                (d.models || []).map(m=>m[0] + " " + Number(m[1]).toFixed(2)).join(" · ") :
                "| <span title='auto-label: track mean score / this frame'>score " +
                (d.score !== undefined ? Number(d.score).toFixed(2) : "-") + "/" +
                (d.interpolated ? "插值" : (d.det_score !== undefined ? Number(d.det_score).toFixed(2) : "-"))) +
                "</span> | <span title='auto-label speed'>" + (d.moving ? "运动 " : "静止 ") +
                Number(d.speed || 0).toFixed(2) + " m/s</span>" +
                (d.points !== undefined ? " | <span title='points in the current sweep / all 5 sweeps'>pts " +
                 d.points + "/" + d.points_all + "</span>" : "") +
                (d.train_as ? " | <span title='what training makes of it without review'>" + d.train_as + "</span>" : "");
        }
        if (box.follows){
            this.boxUi.innerHTML += "| F:"+box.follows.obj_track_id;
        }
    },

    this.set_ref_obj = function(marked_object){
        this.refObjUi.innerHTML="| Ref: "+marked_object.scene+"/"+marked_object.frame+": "+marked_object.ann.obj_type+"-"+marked_object.ann.obj_id;
    },

    this.set_frame_info =function(scene, frame, on_scene_changed){
        
        if (this.sceneSelectorUi.value != scene){
            this.sceneSelectorUi.value = scene;
            on_scene_changed(scene);
        }

        this.frameSelectorUi.value = frame;
    },

    this.clear_frame_info = function(scene, frame){

    },
    
    this.updateModifiedStatus = function(){
        let frames = this.data.worldList.filter(w=>w.annotation.modified);
        if (frames.length > 0)
        {
            this.ui.querySelector("#changed-mark").className = 'ui-button alarm-mark';            
        }
        else
        {
            this.ui.querySelector("#changed-mark").className = 'ui-button';
        }
    }

    this.ui.querySelector("#changed-mark").onmouseenter = ()=>{
        
        let items = "";
        let frames = this.data.worldList.filter(w=>w.annotation.modified).map(w=>w.frameInfo);
        frames.forEach(f=>{
            items += "<div class='modified-world-item'>" + f.frame + '</div>';
        });

        if (frames.length > 0){
            this.ui.querySelector("#changed-world-list").innerHTML = items;
            this.ui.querySelector("#changed-world-list-wrapper").style.display = 'inherit';
        }
    }

    this.ui.querySelector("#changed-mark").onmouseleave = ()=>{
        this.ui.querySelector("#changed-world-list-wrapper").style.display = 'none';
    }

    this.ui.querySelector("#save-button").onclick = ()=>{
        saveWorldList(this.data.worldList);
    }
};


export {Header}