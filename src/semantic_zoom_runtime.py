"""HTML and browser runtime for the semantic-zoom visualization."""

from __future__ import annotations

import html
import json


def render_html(article: str, spec: dict, style: dict[str, object]) -> str:
    BACKGROUND_COLOR = style["BACKGROUND_COLOR"]
    EDGE_DETAILED_LABEL_OFFSET = style["EDGE_DETAILED_LABEL_OFFSET"]
    EDGE_LABEL_REPEAT_DISTANCE = style["EDGE_LABEL_REPEAT_DISTANCE"]
    EDGE_SHORT_LABEL_OFFSET = style["EDGE_SHORT_LABEL_OFFSET"]
    EXACT_REVERT_COLOR = style["EXACT_REVERT_COLOR"]
    LABEL_COARSER_DEPTH_PENALTY = style["LABEL_COARSER_DEPTH_PENALTY"]
    LABEL_COLLISION_PADDING = style["LABEL_COLLISION_PADDING"]
    LABEL_DETAIL_DENSITY_ENTER = style["LABEL_DETAIL_DENSITY_ENTER"]
    LABEL_DETAIL_DENSITY_EXIT = style["LABEL_DETAIL_DENSITY_EXIT"]
    LABEL_DETAIL_MIN_ZOOM_LEVEL = style["LABEL_DETAIL_MIN_ZOOM_LEVEL"]
    LABEL_EDGE_DENSITY_ENTER = style["LABEL_EDGE_DENSITY_ENTER"]
    LABEL_EDGE_DENSITY_EXIT = style["LABEL_EDGE_DENSITY_EXIT"]
    LABEL_FINER_DEPTH_PENALTY = style["LABEL_FINER_DEPTH_PENALTY"]
    LABEL_OBSTACLE_GRID_SIZE = style["LABEL_OBSTACLE_GRID_SIZE"]
    LABEL_SPATIAL_HASH_CELL_SIZE = style["LABEL_SPATIAL_HASH_CELL_SIZE"]
    MIN_DETAILED_EDGE_SCREEN_LENGTH = style["MIN_DETAILED_EDGE_SCREEN_LENGTH"]
    NOISE_COLOR = style["NOISE_COLOR"]
    PARENT_LABEL_DATA_PADDING = style["PARENT_LABEL_DATA_PADDING"]
    PARENT_LABEL_RADIAL_GAP = style["PARENT_LABEL_RADIAL_GAP"]
    REVERT_ARC_STROKE_OPACITY = style["REVERT_ARC_STROKE_OPACITY"]
    REVERT_ARC_STROKE_WIDTH = style["REVERT_ARC_STROKE_WIDTH"]
    TRAJECTORY_COLOR = style["TRAJECTORY_COLOR"]
    WORD_CHANGE_COLOURS = style["WORD_CHANGE_COLOURS"]
    WORD_CHANGE_LABELS = style["WORD_CHANGE_LABELS"]
    encoded = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
    word_change_legend = "".join(
        (
            '<li><span class="legend-dot" style="--legend-color:'
            f'{html.escape(WORD_CHANGE_COLOURS[change_type])}'
            '"></span><span>'
            f'{html.escape(WORD_CHANGE_LABELS[change_type])}'
            "</span></li>"
        )
        for change_type in ("start", "added", "removed", "mixed", "no_change")
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meaningful Projection Space</title>
<script src="https://cdn.jsdelivr.net/npm/vega@5.33.1"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-tooltip@0.35.2/build/vega-tooltip.min.js"></script>
<style>
html,body{{height:100%;margin:0;background:{BACKGROUND_COLOR};color:#151915;font-family:Inter,Arial,sans-serif}}
#app{{width:min(100%,1200px);height:min(100%,720px);margin:0 auto;display:flex;flex-direction:column;overflow:hidden;box-sizing:border-box}}
#topbar{{padding:14px 20px 8px;display:flex;align-items:center;gap:22px;flex-wrap:wrap}}
#topbar h1{{font-size:20px;line-height:1.2;font-weight:700;margin:0}}
#content{{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 250px;border-top:1px solid #e2e5df}}
#vis{{flex:1;min-width:0;min-height:0;overflow:hidden}}
#vis svg{{cursor:grab;overflow:hidden}}
#vis svg:active{{cursor:grabbing}}
#vis svg .semanticLabels text,#vis svg .edgeLabels text{{paint-order:stroke fill;stroke-linejoin:round;transition:opacity 150ms ease}}
#legend{{min-width:0;overflow:auto;border-left:1px solid #d9ddd6;padding:16px 16px 22px;background:#f7f6f2;box-sizing:border-box}}
#legend h2{{font-size:17px;margin:0 0 15px}}
#legend h3{{font-size:12px;line-height:1.2;letter-spacing:.06em;text-transform:uppercase;color:#687068;margin:18px 0 9px}}
.legend-list{{list-style:none;margin:0;padding:0;display:grid;gap:8px}}
.legend-list li{{min-width:0;display:flex;align-items:center;gap:8px;font-size:12px;line-height:1.25}}
.legend-dot{{--legend-color:{NOISE_COLOR};width:14px;height:14px;box-sizing:border-box;border-radius:50%;background:var(--legend-color);border:0;box-shadow:none;flex:none}}
.legend-line{{width:30px;height:0;border-top:2.25px solid {TRAJECTORY_COLOR};flex:none}}
.legend-line.thick{{border-top-width:4.45px}}
.legend-arc{{width:30px;height:10px;box-sizing:border-box;border:{REVERT_ARC_STROKE_WIDTH}px dotted {EXACT_REVERT_COLOR};border-bottom:0;border-radius:50% 50% 0 0;flex:none;opacity:{REVERT_ARC_STROKE_OPACITY}}}
.legend-arrow{{width:30px;flex:none;text-align:center;color:{TRAJECTORY_COLOR};font-size:23px;font-weight:800;line-height:10px}}
@media(max-width:760px){{#content{{grid-template-columns:minmax(0,1fr) 205px}}#legend{{padding:12px 10px}}}}
</style></head><body><main id="app">
<header id="topbar"><h1>Meaningful Projection Space</h1></header>
<div id="content">
<div id="vis" role="img" aria-label="Interactive semantic revision timecurve"></div>
<aside id="legend" aria-label="Visualization legend">
<h2>Legend</h2>
<h3>Display</h3>
<ul class="legend-list">
<li><span class="legend-arrow">&#8594;</span><span>Revision direction</span></li>
<li><span class="legend-line thick"></span><span>Repeated projected path</span></li>
<li><span class="legend-arc"></span><span>Exact revert return</span></li>
</ul>
<h3>Revision changes</h3>
<ul class="legend-list">{word_change_legend}</ul>
</aside>
</div>
</main>
<script>
const spec={encoded};
const handler=new vegaTooltip.Handler({{theme:'light'}});
const vis=document.getElementById('vis');
const view=new vega.View(vega.parse(spec),{{renderer:'svg',hover:true}})
  .initialize(vis).tooltip(handler.call);
let resizeFrame=0;
let visibilityTimer=0;
let visibilitySyncing=false;
let visibilityPending=false;
let lastWidth=0;
let lastHeight=0;
let previousAcceptedParentIds=new Set();
let previousAcceptedEdgeIds=new Set();
const parentPlacementCache=new Map();
const edgePlacementCache=new Map();
function sameIds(first,second){{
  return first.length===second.length&&first.every((value,index)=>value===second[index]);
}}
function normalizedLabelKey(value){{
  return String(value||'').toLocaleLowerCase('en')
    .replace(/[^a-z0-9]+/g,' ').trim();
}}
function renderedLabelItems(mode,...markNames){{
  const names=new Set(markNames);
  const blockedParents=new Set(view.signal('blockedParentLabelIds')||[]);
  const blockedEdges=new Set(view.signal('blockedEdgeLabelIds')||[]);
  const items=new Map();
  function visit(item){{
    if(!item)return;
    if(item.mark&&names.has(item.mark.name)&&item.bounds){{
      const datum=item.datum&&item.datum.datum?item.datum.datum:item.datum;
      const bounds=item.bounds;
      if(datum&&datum.node_id&&
         Number.isFinite(bounds.x1)&&Number.isFinite(bounds.y1)&&
         Number.isFinite(bounds.x2)&&Number.isFinite(bounds.y2)){{
        const markName=item.mark.name;
        const kind=markName==='semanticLabels'?'parent':'edge';
        const variant=markName==='edgeDetailedMeasureLabels'
          ? 'detailed'
          : markName==='edgeShortMeasureLabels'
            ? 'short'
            : 'visible';
        const id=String(datum.node_id);
        const level=String(datum.level||'');
        const collisionBlocked=kind==='parent'
          ? blockedParents.has(id)
          : blockedEdges.has(id);
        // Collision-hidden marks stay in the scenegraph and remain measurable.
        // They can therefore return immediately when zooming opens enough room.
        const include=mode==='collision'||!collisionBlocked;
        if(!include){{
          if(Array.isArray(item.items))item.items.forEach(visit);
          return;
        }}
        items.set(`${{kind}}:${{variant}}:${{id}}`,{{
          id,
          kind,
          variant,
          level,
          hasDetailed:Boolean(String(datum.detailed_description||'').trim()),
          parentId:datum.parent_id==null?'':String(datum.parent_id),
          ancestorIds:Array.isArray(datum.ancestor_ids)
            ? datum.ancestor_ids.map(String)
            : [],
          supportPointIndices:Array.isArray(datum.support_point_indices)
            ? datum.support_point_indices.map(Number)
            : [],
          labelKey:normalizedLabelKey(datum.label),
          priority:Number(datum.label_priority)||0,
          importance:Number(datum.importance)||0,
          parentPlacementDx:Number(datum.parentPlacementDx)||0,
          parentPlacementDy:Number(datum.parentPlacementDy)||0,
          edgeScreenLength:Number(datum.screenLength)||0,
          screenX1:Number(datum.screenX1),
          screenY1:Number(datum.screenY1),
          screenC1X:Number(datum.screenC1X),
          screenC1Y:Number(datum.screenC1Y),
          screenC2X:Number(datum.screenC2X),
          screenC2Y:Number(datum.screenC2Y),
          screenX2:Number(datum.screenX2),
          screenY2:Number(datum.screenY2),
          clipT0:Number(datum.clipT0),
          clipT1:Number(datum.clipT1),
          x1:bounds.x1,
          y1:bounds.y1,
          x2:bounds.x2,
          y2:bounds.y2
        }});
      }}
    }}
    if(Array.isArray(item.items))item.items.forEach(visit);
  }}
  visit(view.scenegraph().root);
  return Array.from(items.values());
}}
function renderedLabelIds(...markNames){{
  return renderedLabelItems('visible',...markNames)
    .map(item=>item.id)
    .sort();
}}
const labelCollisionPadding={LABEL_COLLISION_PADDING};
const labelSpatialHashCellSize={LABEL_SPATIAL_HASH_CELL_SIZE};
const parentLabelRadialGap={PARENT_LABEL_RADIAL_GAP};
const parentLabelDataPadding={PARENT_LABEL_DATA_PADDING};
const edgeLabelRepeatDistance={EDGE_LABEL_REPEAT_DISTANCE};
const labelDetailMinZoomLevel={LABEL_DETAIL_MIN_ZOOM_LEVEL};
const labelObstacleGridSize={LABEL_OBSTACLE_GRID_SIZE};
const edgeDensityEnter={LABEL_EDGE_DENSITY_ENTER};
const edgeDensityExit={LABEL_EDGE_DENSITY_EXIT};
const detailDensityEnter={LABEL_DETAIL_DENSITY_ENTER};
const detailDensityExit={LABEL_DETAIL_DENSITY_EXIT};
function boxesOverlap(first,second,padding=labelCollisionPadding){{
  return first.x1-padding<=second.x2+padding&&
    first.x2+padding>=second.x1-padding&&
    first.y1-padding<=second.y2+padding&&
    first.y2+padding>=second.y1-padding;
}}
function outsideViewport(candidate,padding=4){{
  return candidate.x1-padding<0||candidate.y1-padding<0||
    candidate.x2+padding>view.width()||candidate.y2+padding>view.height();
}}
class LabelSpatialHash{{
  constructor(cellSize=labelSpatialHashCellSize){{
    this.cellSize=cellSize;
    this.cells=new Map();
  }}
  bounds(box){{
    return {{
      minX:Math.floor((box.x1-labelCollisionPadding)/this.cellSize),
      maxX:Math.floor((box.x2+labelCollisionPadding)/this.cellSize),
      minY:Math.floor((box.y1-labelCollisionPadding)/this.cellSize),
      maxY:Math.floor((box.y2+labelCollisionPadding)/this.cellSize)
    }};
  }}
  overlaps(box){{
    const b=this.bounds(box);
    for(let x=b.minX;x<=b.maxX;x+=1){{
      for(let y=b.minY;y<=b.maxY;y+=1){{
        const bucket=this.cells.get(`${{x}},${{y}}`);
        if(bucket&&bucket.some(other=>boxesOverlap(box,other)))return true;
      }}
    }}
    return false;
  }}
  insert(box){{
    const b=this.bounds(box);
    for(let x=b.minX;x<=b.maxX;x+=1){{
      for(let y=b.minY;y<=b.maxY;y+=1){{
        const key=`${{x}},${{y}}`;
        const bucket=this.cells.get(key);
        if(bucket)bucket.push(box);else this.cells.set(key,[box]);
      }}
    }}
  }}
}}
class ObstacleDensityGrid{{
  constructor(cellSize=labelObstacleGridSize){{
    this.cellSize=cellSize;
    this.cells=new Map();
  }}
  add(x,y,weight=1,radius=0){{
    if(!Number.isFinite(x)||!Number.isFinite(y))return;
    const minX=Math.floor((x-radius)/this.cellSize);
    const maxX=Math.floor((x+radius)/this.cellSize);
    const minY=Math.floor((y-radius)/this.cellSize);
    const maxY=Math.floor((y+radius)/this.cellSize);
    for(let cellX=minX;cellX<=maxX;cellX+=1){{
      for(let cellY=minY;cellY<=maxY;cellY+=1){{
        const centerX=(cellX+0.5)*this.cellSize;
        const centerY=(cellY+0.5)*this.cellSize;
        const reach=radius+this.cellSize*0.72;
        if(Math.hypot(centerX-x,centerY-y)>reach)continue;
        const key=`${{cellX}},${{cellY}}`;
        this.cells.set(key,(this.cells.get(key)||0)+weight);
      }}
    }}
  }}
  score(box){{
    const minX=Math.floor(box.x1/this.cellSize);
    const maxX=Math.floor(box.x2/this.cellSize);
    const minY=Math.floor(box.y1/this.cellSize);
    const maxY=Math.floor(box.y2/this.cellSize);
    let total=0;
    let occupied=0;
    let weighted=0;
    let maximum=0;
    for(let cellX=minX;cellX<=maxX;cellX+=1){{
      for(let cellY=minY;cellY<=maxY;cellY+=1){{
        total+=1;
        const value=this.cells.get(`${{cellX}},${{cellY}}`)||0;
        if(value>0)occupied+=1;
        weighted+=Math.min(value,4)/4;
        maximum=Math.max(maximum,value);
      }}
    }}
    if(!total)return 1;
    const occupiedShare=occupied/total;
    const weightedShare=weighted/total;
    const peak=Math.min(maximum,4)/4;
    return 0.55*occupiedShare+0.30*weightedShare+0.15*peak;
  }}
}}
function parentSupportScreenPoints(base){{
  const points=view.data('points')||[];
  const xScale=view.scale('xscale');
  const yScale=view.scale('yscale');
  return base.supportPointIndices.map(index=>points[index])
    .filter(Boolean)
    .map(point=>({{x:xScale(point.x),y:yScale(point.y)}}))
    .filter(point=>Number.isFinite(point.x)&&Number.isFinite(point.y));
}}
function parentPlacementOptions(base){{
  const width=Math.max(1,base.x2-base.x1);
  const height=Math.max(1,base.y2-base.y1);
  const horizontal=Math.min(160,Math.max(24,width*0.5+parentLabelRadialGap));
  const vertical=Math.min(56,Math.max(18,height+parentLabelRadialGap));
  const currentDx=Number(base.parentPlacementDx)||0;
  const currentDy=Number(base.parentPlacementDy)||0;
  const baseCenterX=(base.x1+base.x2)/2-currentDx;
  const baseCenterY=(base.y1+base.y2)/2-currentDy;
  const cached=parentPlacementCache.get(base.id);
  const options=[];
  if(cached)options.push({{dx:cached.dx,dy:cached.dy,cached:true}});
  let variable;
  if(base.level==='root'){{
    variable=[{{dx:0,dy:0}}];
  }}else if(['macro','cluster','section'].includes(base.level)){{
    const support=parentSupportScreenPoints(base);
    if(support.length){{
      const centroid={{
        x:support.reduce((sum,point)=>sum+point.x,0)/support.length,
        y:support.reduce((sum,point)=>sum+point.y,0)/support.length
      }};
      const centralSupport=support.slice().sort((first,second)=>
        Math.hypot(first.x-centroid.x,first.y-centroid.y)-
        Math.hypot(second.x-centroid.x,second.y-centroid.y)
      ).slice(0,8);
      variable=[centroid,...centralSupport].map(point=>({{
        dx:point.x-baseCenterX,
        dy:point.y-baseCenterY
      }}));
      variable.unshift({{dx:0,dy:0}});
    }}else{{
      variable=[{{dx:0,dy:0}}];
    }}
  }}else{{
    variable=[
        {{dx:0,dy:0}},{{dx:0,dy:-vertical}},{{dx:0,dy:vertical}},
        {{dx:horizontal,dy:0}},{{dx:-horizontal,dy:0}},
        {{dx:horizontal*0.72,dy:-vertical*0.78}},
        {{dx:-horizontal*0.72,dy:-vertical*0.78}},
        {{dx:horizontal*0.72,dy:vertical*0.78}},
        {{dx:-horizontal*0.72,dy:vertical*0.78}}
      ];
  }}
  for(const option of variable){{
    if(!options.some(item=>
      Math.abs(item.dx-option.dx)<0.1&&Math.abs(item.dy-option.dy)<0.1
    ))options.push({{...option,cached:false}});
  }}
  return options;
}}
function parentCandidateAt(base,option){{
  const currentDx=Number(base.parentPlacementDx)||0;
  const currentDy=Number(base.parentPlacementDy)||0;
  return {{
    ...base,
    placementDx:option.dx,
    placementDy:option.dy,
    x1:base.x1-currentDx+option.dx,
    y1:base.y1-currentDy+option.dy,
    x2:base.x2-currentDx+option.dx,
    y2:base.y2-currentDy+option.dy
  }};
}}
function parentSupportCoverage(base,candidate){{
  const support=parentSupportScreenPoints(base);
  let count=0;
  for(const point of support){{
    if(point.x>=candidate.x1-parentLabelDataPadding&&
       point.x<=candidate.x2+parentLabelDataPadding&&
       point.y>=candidate.y1-parentLabelDataPadding&&
       point.y<=candidate.y2+parentLabelDataPadding)count+=1;
  }}
  if(count||base.supportPointIndices.length)return count;
  // Compatibility fallback for externally supplied hierarchies without
  // explicit membership: the reviewed semantic anchor itself is the support.
  const anchorX=(base.x1+base.x2)/2-(Number(base.parentPlacementDx)||0);
  const anchorY=(base.y1+base.y2)/2-(Number(base.parentPlacementDy)||0);
  return anchorX>=candidate.x1-parentLabelDataPadding&&
    anchorX<=candidate.x2+parentLabelDataPadding&&
    anchorY>=candidate.y1-parentLabelDataPadding&&
    anchorY<=candidate.y2+parentLabelDataPadding ? 1 : 0;
}}
function bestParentPlacement(base,occupied,obstacles){{
  let best=null;
  let rank=0;
  for(const option of parentPlacementOptions(base)){{
    const candidate=parentCandidateAt(base,option);
    if(outsideViewport(candidate,8)||occupied.overlaps(candidate)){{
      rank+=1;
      continue;
    }}
    const supportCoverage=parentSupportCoverage(base,candidate);
    if(supportCoverage<1){{
      rank+=1;
      continue;
    }}
    const density=obstacles.score(candidate);
    const displacement=Math.hypot(option.dx,option.dy);
    const supportReward=Math.min(6,supportCoverage)*0.08;
    const score=density*0.08+Math.min(0.08,displacement/2400)+
      rank*0.002-supportReward;
    if(option.cached)return {{...candidate,density,supportCoverage,score}};
    if(!best||score<best.score){{
      best={{...candidate,density,supportCoverage,score}};
    }}
    rank+=1;
  }}
  return best;
}}
function cubicPoint(edge,t){{
  const u=1-t;
  return {{
    x:u*u*u*edge.screenX1+3*u*u*t*edge.screenC1X+
      3*u*t*t*edge.screenC2X+t*t*t*edge.screenX2,
    y:u*u*u*edge.screenY1+3*u*u*t*edge.screenC1Y+
      3*u*t*t*edge.screenC2Y+t*t*t*edge.screenY2
  }};
}}
function cubicTangent(edge,t){{
  const u=1-t;
  return {{
    x:3*u*u*(edge.screenC1X-edge.screenX1)+
      6*u*t*(edge.screenC2X-edge.screenC1X)+
      3*t*t*(edge.screenX2-edge.screenC2X),
    y:3*u*u*(edge.screenC1Y-edge.screenY1)+
      6*u*t*(edge.screenC2Y-edge.screenC1Y)+
      3*t*t*(edge.screenY2-edge.screenC2Y)
  }};
}}
function buildObstacleDensityGrid(){{
  const grid=new ObstacleDensityGrid();
  for(const edge of view.data('splineEdgeGeometry')||[]){{
    const weight=1+0.22*Math.min(4,Math.max(0,
      Number(edge.overlap_count||1)-1));
    for(let step=0;step<=18;step+=1){{
      const point=cubicPoint(edge,step/18);
      grid.add(point.x,point.y,weight,2.5);
    }}
  }}
  for(const arc of view.data('revertArcGeometry')||[]){{
    for(let step=0;step<=14;step+=1){{
      const t=step/14;
      const u=1-t;
      grid.add(
        u*u*arc.screenX1+2*u*t*arc.controlX+t*t*arc.screenX2,
        u*u*arc.screenY1+2*u*t*arc.controlY+t*t*arc.screenY2,
        0.8,2
      );
    }}
  }}
  const xScale=view.scale('xscale');
  const yScale=view.scale('yscale');
  for(const point of view.data('points')||[]){{
    grid.add(xScale(point.x),yScale(point.y),3.5,7);
  }}
  return grid;
}}
function candidateAt(base,t,side,offset){{
  const point=cubicPoint(base,t);
  const tangent=cubicTangent(base,t);
  const length=Math.hypot(tangent.x,tangent.y);
  const unitX=side*(length>=1 ? -tangent.y/length : 1);
  const unitY=side*(length>=1 ? tangent.x/length : 0);
  const width=Math.max(1,base.x2-base.x1);
  const height=Math.max(1,base.y2-base.y1);
  const centerX=point.x+offset*unitX;
  const centerY=point.y+offset*unitY;
  return {{
    ...base,t,side,centerX,centerY,
    x1:centerX-width/2,y1:centerY-height/2,
    x2:centerX+width/2,y2:centerY+height/2
  }};
}}
function placementOptions(base){{
  const start=Math.max(0,Math.min(1,base.clipT0));
  const end=Math.max(start,Math.min(1,base.clipT1));
  const span=end-start;
  const cached=edgePlacementCache.get(base.id);
  const fractions=[0.5,0.35,0.65,0.2,0.8];
  const positions=[];
  if(cached&&cached.t>=start&&cached.t<=end){{
    positions.push({{t:cached.t,side:cached.side,cached:true}});
  }}
  for(const fraction of fractions){{
    const t=start+span*fraction;
    for(const side of [1,-1]){{
      if(!positions.some(item=>Math.abs(item.t-t)<1e-6&&item.side===side)){{
        positions.push({{t,side,cached:false}});
      }}
    }}
  }}
  return positions;
}}
function bestEdgePlacement(base,offset,occupied,obstacles,wasVisible,
                           enterThreshold,exitThreshold,
                           placementAllowed=()=>true){{
  const threshold=wasVisible?exitThreshold:enterThreshold;
  let best=null;
  for(const option of placementOptions(base)){{
    const candidate=candidateAt(base,option.t,option.side,offset);
    if(outsideViewport(candidate,8)||occupied.overlaps(candidate)||
       !placementAllowed(candidate))continue;
    const density=obstacles.score(candidate);
    if(density>threshold)continue;
    const distancePenalty=Math.abs(option.t-0.5)*0.06;
    const sidePenalty=option.side<0?0.004:0;
    const score=density+distancePenalty+sidePenalty;
    if(option.cached)return {{...candidate,density}};
    if(!best||score<best.score)best={{...candidate,density,score}};
  }}
  return best;
}}
function edgeLabelRepeatsNearby(acceptedPositions,labelKey,candidate){{
  if(!labelKey)return false;
  const positions=acceptedPositions.get(labelKey)||[];
  return positions.some(position=>
    Math.hypot(position.x-candidate.centerX,position.y-candidate.centerY)<
      edgeLabelRepeatDistance
  );
}}
function registerEdgeLabelPosition(acceptedPositions,labelKey,candidate){{
  if(!labelKey)return;
  const positions=acceptedPositions.get(labelKey)||[];
  positions.push({{x:candidate.centerX,y:candidate.centerY}});
  acceptedPositions.set(labelKey,positions);
}}
function semanticDepth(candidate,variant='visible'){{
  if(candidate.kind==='edge')return variant==='detailed'?5:4;
  if(candidate.level==='root')return 0;
  if(candidate.level==='macro')return 1;
  if(candidate.level==='cluster')return 2;
  if(candidate.level==='section')return 3;
  return 0;
}}
function semanticPlacementScore(candidate,variant,targetDepth){{
  const declaredPriority=Math.min(140,Math.max(0,candidate.priority)*18);
  const importanceBonus=Math.min(
    180,Math.log1p(Math.max(0,candidate.importance))*24
  );
  const wasVisible=candidate.kind==='edge'
    ? previousAcceptedEdgeIds.has(candidate.id)
    : previousAcceptedParentIds.has(candidate.id);
  const stabilityBonus=wasVisible?(candidate.kind==='edge'?220:160):0;
  if(candidate.level==='root'){{
    return 6000+declaredPriority+importanceBonus+stabilityBonus;
  }}
  if(candidate.level==='endpoint'){{
    return 5200+declaredPriority+importanceBonus+stabilityBonus;
  }}
  const depth=semanticDepth(candidate,variant);
  const depthDelta=depth-targetDepth;
  // Finer-than-current labels are delayed more strongly than useful parent
  // context, but they remain candidates and appear whenever space is free.
  const distancePenalty=Math.abs(depthDelta)*(
    depthDelta>0?{LABEL_FINER_DEPTH_PENALTY}:{LABEL_COARSER_DEPTH_PENALTY}
  );
  // Unlike SHACLens, this timecurve treats revision edges as first-class
  // semantic content. Their priority rises continuously from section detail
  // toward short and finally full descriptions as the user zooms in.
  const edgeZoomBonus=candidate.kind==='edge'
    ? Math.max(0,targetDepth-3)*180
    : 0;
  return 3000-distancePenalty+declaredPriority+importanceBonus+
    stabilityBonus+edgeZoomBonus;
}}
function collisionBlocks(){{
  const compareCandidates=(first,second)=>{{
    const priorityDifference=second.priority-first.priority;
    if(priorityDifference)return priorityDifference;
    const importanceDifference=second.importance-first.importance;
    if(importanceDifference)return importanceDifference;
    return `${{first.kind}}:${{first.id}}`.localeCompare(`${{second.kind}}:${{second.id}}`);
  }};
  const parents=renderedLabelItems('collision','semanticLabels')
    .sort(compareCandidates);
  const shortEdges=renderedLabelItems(
    'collision','edgeShortMeasureLabels'
  ).sort(compareCandidates);
  const detailedEdges=new Map(
    renderedLabelItems('collision','edgeDetailedMeasureLabels')
      .map(candidate=>[candidate.id,candidate])
  );
  const zoomLevel=Math.max(0,Number(view.signal('zoomLevel'))||0);
  const targetDepth=Math.max(
    0,Math.min(5,Number(view.signal('semanticTargetDepth'))||1)
  );
  const occupied=new LabelSpatialHash();
  const obstacles=buildObstacleDensityGrid();
  const acceptedEdgeLabelPositions=new Map();
  const acceptedParentIds=new Set();
  const acceptedEdgeIds=new Set();
  const parentPlacements=new Map();
  const placements=new Map();
  const blockedParents=[];
  const blockedEdges=[];
  const detailedEdgeIds=[];
  const acceptCandidate=candidate=>{{
    occupied.insert(candidate);
  }};
  const edgeActions=shortEdges.map(shortCandidate=>{{
    const detailedCandidate=detailedEdges.get(shortCandidate.id);
    const detailedWidth=detailedCandidate
      ? Math.max(0,detailedCandidate.x2-detailedCandidate.x1)
      : 0;
    const requiredDetailLength=Math.max(
      {MIN_DETAILED_EDGE_SCREEN_LENGTH},Math.min(240,detailedWidth*0.55)
    );
    const detailedEligible=zoomLevel>=labelDetailMinZoomLevel&&
      detailedCandidate&&detailedCandidate.hasDetailed&&
      shortCandidate.edgeScreenLength>=requiredDetailLength;
    const priorityCandidate=detailedEligible
      ? detailedCandidate
      : shortCandidate;
    const variant=detailedEligible?'detailed':'short';
    return {{
      kind:'edge',candidate:shortCandidate,detailedCandidate,
      detailedEligible,score:semanticPlacementScore(
        priorityCandidate,variant,targetDepth
      )
    }};
  }});
  const actions=parents.map(candidate=>({{
    kind:'parent',candidate,
    score:semanticPlacementScore(candidate,'visible',targetDepth)
  }})).concat(edgeActions).sort((first,second)=>{{
    const scoreDifference=second.score-first.score;
    if(scoreDifference)return scoreDifference;
    return compareCandidates(first.candidate,second.candidate);
  }});
  // SHACLens-style greedy screen-space placement, adapted to make edge labels
  // the preferred detailed layer. No semantic level is hard-disabled: coarse
  // labels win while zoomed out, edge descriptions win close up, and every
  // lower-priority label is still accepted when its actual screen box fits.
  for(const action of actions){{
    if(action.kind==='parent'){{
      const candidate=action.candidate;
      const parentPlacement=bestParentPlacement(
        candidate,occupied,obstacles
      );
      if(!parentPlacement){{
        blockedParents.push(candidate.id);
      }}else{{
        acceptCandidate(parentPlacement);
        acceptedParentIds.add(candidate.id);
        parentPlacements.set(candidate.id,{{
          dx:parentPlacement.placementDx,
          dy:parentPlacement.placementDy
        }});
      }}
      continue;
    }}
    const shortCandidate=action.candidate;
    const detailedCandidate=action.detailedCandidate;
    const detailedEligible=action.detailedEligible;
    const wasVisible=previousAcceptedEdgeIds.has(shortCandidate.id);
    const hasRepeatSpace=placement=>!edgeLabelRepeatsNearby(
      acceptedEdgeLabelPositions,shortCandidate.labelKey,placement
    );
    const detailedPlacement=detailedEligible
      ? bestEdgePlacement(
          detailedCandidate,{EDGE_DETAILED_LABEL_OFFSET},occupied,obstacles,
          wasVisible,detailDensityEnter,detailDensityExit,hasRepeatSpace
        )
      : null;
    // Prefer the concrete full revision description whenever zooming creates
    // enough contextual screen space. Repeated short text is permitted again
    // after a map-like minimum screen distance instead of being hidden
    // globally across the complete trajectory.
    if(detailedPlacement){{
      acceptCandidate(detailedPlacement);
      detailedEdgeIds.push(shortCandidate.id);
      acceptedEdgeIds.add(shortCandidate.id);
      placements.set(shortCandidate.id,{{
        t:detailedPlacement.t,side:detailedPlacement.side
      }});
      registerEdgeLabelPosition(
        acceptedEdgeLabelPositions,shortCandidate.labelKey,detailedPlacement
      );
    }}else{{
      const shortPlacement=bestEdgePlacement(
        shortCandidate,{EDGE_SHORT_LABEL_OFFSET},occupied,obstacles,
        wasVisible,edgeDensityEnter,edgeDensityExit,hasRepeatSpace
      );
      if(shortPlacement){{
        acceptCandidate(shortPlacement);
        acceptedEdgeIds.add(shortCandidate.id);
        placements.set(shortCandidate.id,{{
          t:shortPlacement.t,side:shortPlacement.side
        }});
        registerEdgeLabelPosition(
          acceptedEdgeLabelPositions,shortCandidate.labelKey,shortPlacement
        );
      }}else{{
        blockedEdges.push(shortCandidate.id);
      }}
    }}
  }}
  const parentPlacementRows=Array.from(parentPlacements.entries())
    .sort((first,second)=>first[0].localeCompare(second[0]));
  const placementRows=Array.from(placements.entries())
    .sort((first,second)=>first[0].localeCompare(second[0]));
  return {{
    parentIds:Array.from(new Set(blockedParents)).sort(),
    edgeIds:Array.from(new Set(blockedEdges)).sort(),
    detailedEdgeIds:Array.from(new Set(detailedEdgeIds)).sort(),
    acceptedParentIds:Array.from(acceptedParentIds).sort(),
    acceptedEdgeIds:Array.from(acceptedEdgeIds).sort(),
    parentPlacementIds:parentPlacementRows.map(row=>row[0]),
    parentPlacementDx:parentPlacementRows.map(row=>row[1].dx),
    parentPlacementDy:parentPlacementRows.map(row=>row[1].dy),
    placementIds:placementRows.map(row=>row[0]),
    placementTs:placementRows.map(row=>row[1].t),
    placementSides:placementRows.map(row=>row[1].side)
  }};
}}
async function syncLabelVisibility(){{
  if(view.signal('down'))return;
  if(visibilitySyncing){{
    visibilityPending=true;
    return;
  }}
  visibilitySyncing=true;
  try{{
    const blocked=collisionBlocks();
    let currentBlockedParents=(view.signal('blockedParentLabelIds')||[]).slice().sort();
    let currentBlockedEdges=(view.signal('blockedEdgeLabelIds')||[]).slice().sort();
    let currentDetailedEdges=(view.signal('detailedEdgeLabelIds')||[]).slice().sort();
    let currentParentPlacementIds=(view.signal('parentLabelPlacementIds')||[]).slice();
    let currentParentPlacementDx=(view.signal('parentLabelPlacementDx')||[]).slice();
    let currentParentPlacementDy=(view.signal('parentLabelPlacementDy')||[]).slice();
    let currentPlacementIds=(view.signal('edgeLabelPlacementIds')||[]).slice();
    let currentPlacementTs=(view.signal('edgeLabelPlacementTs')||[]).slice();
    let currentPlacementSides=(view.signal('edgeLabelPlacementSides')||[]).slice();
    if(!sameIds(blocked.parentIds,currentBlockedParents)||
       !sameIds(blocked.edgeIds,currentBlockedEdges)||
       !sameIds(blocked.detailedEdgeIds,currentDetailedEdges)||
       !sameIds(blocked.parentPlacementIds,currentParentPlacementIds)||
       !sameIds(blocked.parentPlacementDx,currentParentPlacementDx)||
       !sameIds(blocked.parentPlacementDy,currentParentPlacementDy)||
       !sameIds(blocked.placementIds,currentPlacementIds)||
       !sameIds(blocked.placementTs,currentPlacementTs)||
       !sameIds(blocked.placementSides,currentPlacementSides)){{
      view.signal('blockedParentLabelIds',blocked.parentIds);
      view.signal('blockedEdgeLabelIds',blocked.edgeIds);
      view.signal('detailedEdgeLabelIds',blocked.detailedEdgeIds);
      view.signal('parentLabelPlacementIds',blocked.parentPlacementIds);
      view.signal('parentLabelPlacementDx',blocked.parentPlacementDx);
      view.signal('parentLabelPlacementDy',blocked.parentPlacementDy);
      view.signal('edgeLabelPlacementIds',blocked.placementIds);
      view.signal('edgeLabelPlacementTs',blocked.placementTs);
      view.signal('edgeLabelPlacementSides',blocked.placementSides);
      await view.runAsync();
    }}
    previousAcceptedParentIds=new Set(blocked.acceptedParentIds);
    previousAcceptedEdgeIds=new Set(blocked.acceptedEdgeIds);
    for(let index=0;index<blocked.parentPlacementIds.length;index+=1){{
      parentPlacementCache.set(blocked.parentPlacementIds[index],{{
        dx:blocked.parentPlacementDx[index],
        dy:blocked.parentPlacementDy[index]
      }});
    }}
    for(let index=0;index<blocked.placementIds.length;index+=1){{
      edgePlacementCache.set(blocked.placementIds[index],{{
        t:blocked.placementTs[index],
        side:blocked.placementSides[index]
      }});
    }}
    const parentIds=renderedLabelIds('semanticLabels');
    const edgeIds=renderedLabelIds('edgeLabels');
    vis.dataset.zoom=Number(view.signal('zoomLevel')).toFixed(3);
    vis.dataset.visibleLabels=String(parentIds.length+edgeIds.length);
    vis.dataset.visibleHierarchyLabels=String(parentIds.length);
    vis.dataset.visibleEdgeLabels=String(edgeIds.length);
    vis.dataset.blockedParentLabels=String(blocked.parentIds.length);
    vis.dataset.blockedEdgeLabels=String(blocked.edgeIds.length);
    vis.dataset.detailedEdgeLabels=String(blocked.detailedEdgeIds.length);
    vis.dataset.shortEdgeLabels=String(
      Math.max(0,edgeIds.length-blocked.detailedEdgeIds.length)
    );
  }}finally{{
    visibilitySyncing=false;
    if(visibilityPending){{
      visibilityPending=false;
      scheduleLabelVisibility();
    }}
  }}
}}
function scheduleLabelVisibility(){{
  clearTimeout(visibilityTimer);
  visibilityTimer=setTimeout(()=>{{void syncLabelVisibility();}},20);
}}
view.addSignalListener('zooming',(_name,value)=>{{if(!value)scheduleLabelVisibility();}});
view.addSignalListener('zoomLevel',()=>{{
  if(!view.signal('zooming'))scheduleLabelVisibility();
}});
view.addSignalListener('down',(_name,value)=>{{if(value===null)scheduleLabelVisibility();}});
function resizeView(){{
  cancelAnimationFrame(resizeFrame);
  resizeFrame=requestAnimationFrame(async()=>{{
    const width=Math.max(320,vis.clientWidth-56);
    const height=Math.max(420,vis.clientHeight-46);
    if(width===lastWidth&&height===lastHeight)return;
    lastWidth=width;lastHeight=height;
    await view.width(width).height(height).runAsync();
    scheduleLabelVisibility();
  }});
}}
new ResizeObserver(resizeView).observe(vis);
resizeView();
</script></body></html>"""


