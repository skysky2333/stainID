/* Looking at a session's labels (or the reference labels): every labelled object as the crop the model sees around
   it, its outline or size drawn in its class colour, what people said and what detection called it. Filter by class,
   by how detection compares, by field and by person; one object opens larger with everything known about it. */
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import type { AiVerdict, ClassGroup, LabelledObject, SessionObjects } from '../api'
import { api, apiUrl, enc, fieldImage } from '../api'
import { Empty, ErrorNote, Loaded, Loading, Modal, Progress, fmt, timeAgo, withError } from '../components'
import { useFetch } from '../hooks'
import { slotColour } from '../classPalette'
import { useProject } from '../project'
import { Advanced } from '../setup/parts'

const PAGE = 60
const NOTHING = '-'  // a class filter no class matches: every class hidden
type AiFilter = '' | 'disagrees' | 'type' | 'unknown'

/** One colour per identity (its types share it), from the identity's place among the stain's classes, so a class keeps its
 *  colour whatever is shown. */
export const classColor = (groups: ClassGroup[], value: string) => {
  const group = groups.find((g) => g.keys.includes(value))
  return !group || group.colour == null ? 'var(--text-muted)' : slotColour(group.colour)
}

/** A class under its identity's chip, without the identity's name: "pretangle", "type not recorded". */
const typeName = (identity: string, key: string) =>
  key.startsWith(`${identity} · `) ? key.slice(identity.length + 3) : key.startsWith(`${identity} (`) ? key.slice(identity.length + 2, -1) : key

/** Who made a label, in plain words: a person's name, or the earlier label set it was brought in from. */
const who = (source: string) => source.replace(/^import: /, '')

const AI_TEXT: Record<AiVerdict, string> = { agrees: 'Detection agrees', type: 'Detection: other type', disagrees: 'Detection disagrees' }

const cropUrl = (stain: string, o: LabelledObject) => apiUrl(`/api/labels/crops/${enc(stain)}/${enc(o.tile_id)}.jpg?x=${o.x}&y=${o.y}`)

export function SessionLabelsView({ name, title, onClose }: { name: string; title: string; onClose: () => void }) {
  const [stain, setStain] = useState('')
  const [hidden, setHidden] = useState<string[]>([])
  const [ai, setAi] = useState<AiFilter>('')
  const [tile, setTile] = useState('')
  const [person, setPerson] = useState('')
  const [page, setPage] = useState(0)
  const [openAt, setOpenAt] = useState<number | null>(null)
  const [preparing, setPreparing] = useState(false)
  const [known, setKnown] = useState<SessionObjects | null>(null)  // classes of the stain shown, to send the ones not hidden
  const shownClasses = known && hidden.length ? known.vocabulary.filter((c) => known.classes[c] && !hidden.includes(c)) : []
  const values = hidden.length && !shownClasses.length ? NOTHING : shownClasses.join('\n')  // class names may hold commas
  const query = new URLSearchParams({ stain, values, ai, tile, person, offset: String(page * PAGE), limit: String(PAGE) })
  const data = useFetch<SessionObjects>(`/api/labels/sessions/${name}/objects?${query}`, preparing ? 3000 : undefined)
  useEffect(() => {
    if (!data.data) return
    setPreparing(!!data.data.preparing)
    setKnown(data.data)
  }, [data.data])
  const filter = (change: () => void) => { change(); setPage(0); setOpenAt(null) }
  return (
    <Modal title={`Labels of “${title}”`} onClose={onClose} wide>
      <Loaded state={data} label="Loading the labels…">
        {(d) => {
          const crop = d.crop
          const toggle = (keys: string[]) => filter(() => setHidden(keys.every((k) => hidden.includes(k)) ? hidden.filter((h) => !keys.includes(h)) : [...new Set([...hidden, ...keys])]))
          const present = d.groups.map((g) => ({ ...g, keys: g.keys.filter((k) => d.classes[k]) })).filter((g) => g.keys.length)
          return (
            <div className="examine">
              <p className="small secondary" style={{ margin: 0 }}>
                {d.reference ? 'AI reference labels: made before labelling sessions existed, brought in from earlier label sets. ' : ''}
                Each picture is what the model sees around one labelled object{crop ? ` (${fmt(crop.um)} µm across)` : ''}; the ring or outline
                shows the object in the colour of the class people gave it.
              </p>
              {d.stains.length > 1 && (
                <div className="segmented">{d.stains.map((s) => <button key={s} className={`btn small${s === d.stain ? ' chosen' : ''}`} aria-pressed={s === d.stain}
                  onClick={() => filter(() => { setStain(s); setHidden([]); setPerson('') })}>{s}</button>)}</div>
              )}
              <div className="class-groups" role="group" aria-label="Classes shown">
                {present.map((g) => {
                  const total = g.keys.reduce((a, k) => a + d.classes[k], 0)
                  const typed = g.keys.length > 1 || g.keys[0] !== g.name
                  const chip = (key: string, text: string, n: number, keys: string[]) => {
                    const on = !keys.every((k) => hidden.includes(k))
                    return (
                      <button key={key} className={`btn small class-toggle${on ? ' chosen' : ''}`} aria-pressed={on} onClick={() => toggle(keys)}
                        title={on ? `Hide ${text}` : `Show ${text}`}>
                        <span className="swatch round" style={{ background: on ? classColor(d.groups, keys[0]) : 'transparent', borderColor: classColor(d.groups, keys[0]) }} />
                        {text} <span className="muted">{n.toLocaleString()}</span></button>
                    )
                  }
                  return (
                    <div key={g.id} className="class-chips">
                      {chip(g.id, typed ? `${g.name}: all` : g.name, total, g.keys)}
                      {typed && g.keys.map((k) => chip(k, typeName(g.name, k), d.classes[k], [k]))}
                    </div>
                  )
                })}
                {hidden.length > 0 && <button className="link-btn small" onClick={() => filter(() => setHidden([]))}>Show all classes</button>}
              </div>
              <div className="row examine-filters">
                <div className="segmented" role="group" aria-label="Compared with detection">
                  {([['', 'All', d.all], ['disagrees', 'Detection disagrees', d.ai.disagrees ?? 0], ['type', 'Detection gives another type', d.ai.type ?? 0]] as [AiFilter, string, number][])
                    .filter(([key, , n]) => key === '' || n > 0 || ai === key)
                    .map(([key, label, n]) => (
                      <button key={key} className={`btn small${ai === key ? ' chosen' : ''}`} aria-pressed={ai === key} onClick={() => filter(() => setAi(key))}>
                        {label} <span className="muted">{n.toLocaleString()}</span></button>
                    ))}
                </div>
                <span className="spacer" />
                <label className="field inline-field">Field or core<input type="search" value={tile} placeholder="e.g. B-2" style={{ width: 130 }}
                  onChange={(e) => filter(() => setTile(e.target.value))} /></label>
                {Object.keys(d.people).length > 1 && (
                  <label className="field inline-field">Labelled by<select value={person} onChange={(e) => filter(() => setPerson(e.target.value))}>
                    <option value="">anyone</option>
                    {Object.entries(d.people).map(([p, n]) => <option key={p} value={p}>{who(p)} ({n.toLocaleString()})</option>)}
                  </select></label>
                )}
              </div>
              {d.preparing && <PreparingNote preparing={d.preparing} shown={d.objects.length} />}
              {d.objects.length ? <>
                <div className={`object-grid${data.loading ? ' is-loading' : ''}`}>
                  {d.objects.map((o, i) => <ObjectCard key={`${o.tile_id}:${o.x}:${o.y}`} o={o} d={d} onOpen={() => setOpenAt(i)} />)}
                </div>
                <Pages page={page} total={d.total} onPage={(p) => { setPage(p); setOpenAt(null) }} />
              </> : <Empty>
                <p style={{ margin: 0 }}>No labelled objects{hidden.length ? ' of the classes shown' : ''}{ai ? ' where detection differs like that' : ''}{tile ? ` in fields matching “${tile}”` : ''}.</p>
                {(hidden.length > 0 || ai || tile || person) && <button className="btn small" onClick={() => filter(() => { setHidden([]); setAi(''); setTile(''); setPerson('') })}>Show every object</button>}
              </Empty>}
              {openAt !== null && d.objects[openAt] && (
                <ObjectDetail o={d.objects[openAt]} d={d} position={`${page * PAGE + openAt + 1} of ${d.total.toLocaleString()}`}
                  onPrevious={openAt > 0 ? () => setOpenAt(openAt - 1) : undefined}
                  onNext={openAt < d.objects.length - 1 ? () => setOpenAt(openAt + 1) : undefined} onClose={() => setOpenAt(null)} />
              )}
            </div>
          )
        }}
      </Loaded>
    </Modal>
  )
}

function PreparingNote({ preparing, shown }: { preparing: NonNullable<SessionObjects['preparing']>; shown: number }) {
  return (
    <div className="preparing-note small" role="status">
      <Loading inline label={<>Cutting the pictures of {preparing.objects} object{preparing.objects === 1 ? '' : 's'} from {preparing.cores} core
        image{preparing.cores === 1 ? '' : 's'} (a few seconds each{preparing.ahead ? `; ${preparing.ahead} other picture${preparing.ahead === 1 ? '' : 's'} first` : ''}).
        The others are ready to look at.</>} />
      <Progress done={shown - preparing.objects} total={shown} />
    </div>
  )
}

function Pages({ page, total, onPage }: { page: number; total: number; onPage: (page: number) => void }) {
  if (total <= PAGE) return <p className="muted small" style={{ margin: 0 }}>{total.toLocaleString()} object{total === 1 ? '' : 's'}</p>
  return (
    <div className="row picker-pages">
      <button className="btn small" disabled={page === 0} onClick={() => onPage(page - 1)}>‹ Previous</button>
      <span className="small secondary">objects {(page * PAGE + 1).toLocaleString()}–{Math.min(total, (page + 1) * PAGE).toLocaleString()} of {total.toLocaleString()}</span>
      <button className="btn small" disabled={(page + 1) * PAGE >= total} onClick={() => onPage(page + 1)}>Next ›</button>
    </div>
  )
}

/** Where an object is, in a few words ("TMA 3 · B-2 · frontal"), or what its picture is. */
function usePlace() {
  const { tmaName } = useProject()
  return (o: LabelledObject) => {
    if (o.frame === 'pilot') return 'pilot field of an earlier study'
    if (!o.frame) return 'no picture'
    return [o.tma ? tmaName(o.tma) : null, o.core_label, o.region].filter(Boolean).join(' · ')
  }
}

function hoverText(o: LabelledObject, place: string) {
  const labels = o.labels.map((l) => `${l.value} — ${who(l.source)}${l.time ? `, ${timeAgo(l.time)}` : ''}`).join('\n')
  const aiCall = o.ai_value ? `Detection: ${o.ai_value}${o.ai_probability != null ? ` (score ${Math.round(100 * o.ai_probability)}%)` : ''}` : 'Detection: no call here'
  return `${labels}\n${aiCall}\n${place}${o.frame === 'field' ? ` · field ${o.tile_id}` : ''}`
}

function ObjectCard({ o, d, onOpen }: { o: LabelledObject; d: SessionObjects; onOpen: () => void }) {
  const place = usePlace()
  return (
    <button className={`object-card${o.ai === 'disagrees' ? ' ai-disagrees' : ''}`} onClick={onOpen} title={hoverText(o, place(o))}>
      <ObjectPicture o={o} d={d} />
      <span className="object-class small"><span className="swatch round" style={{ background: classColor(d.groups, o.value) }} />
        <span className="ellipsis">{o.value}</span>{o.labels.length > 1 && <span className="muted"> ×{o.labels.length}</span>}</span>
      <AiLine o={o} />
      <span className="small muted ellipsis">{place(o)}</span>
    </button>
  )
}

function AiLine({ o, long = false }: { o: LabelledObject; long?: boolean }) {
  if (!o.ai_value) return <span className="small muted">{long ? 'Detection made no call here (a person added this object).' : 'no detection call'}</span>
  const tone = o.ai === 'disagrees' ? 'ai-no' : o.ai === 'type' ? 'ai-type' : 'ai-yes'
  return (
    <span className={`small ai-line ${tone}`}>
      {o.ai === 'disagrees' && <span className="ai-mark" aria-label="disagrees">≠</span>}
      <span className="ellipsis">{o.ai ? `${long ? AI_TEXT[o.ai] : 'Detection'}: ` : 'Detection: '}{o.ai_value}</span>
      {long && o.ai_probability != null && <span className="muted" title="How likely detection’s model found it that this is a positive object"> · detection score {Math.round(100 * o.ai_probability)}%</span>}
    </span>
  )
}

/** The object's crop with its outline (or a ring of its size) on top; a placeholder while the crop is being cut. */
function ObjectPicture({ o, d, large = false }: { o: LabelledObject; d: SessionObjects; large?: boolean }) {
  const [attempt, setAttempt] = useState(0)
  const [failed, setFailed] = useState(false)
  const size = d.crop?.size_px ?? 160
  useEffect(() => { setFailed(false) }, [o.crop.state])
  if (o.crop.state !== 'ready' || failed) {
    const text = o.crop.state === 'missing' ? 'no picture of this object'
      : o.crop.state === 'failed' ? 'the picture could not be made'
        : o.crop.state === 'waiting' && o.crop.ahead ? `waiting (${o.crop.ahead} ahead)` : 'cutting the picture…'
    return <div className={`object-crop placeholder${large ? ' large' : ''}`}>
      {o.crop.state === 'missing' || o.crop.state === 'failed' ? null : <span className="spinner" aria-hidden="true" />}
      <span className="small muted">{text}</span></div>
  }
  const color = classColor(d.groups, o.value)
  const c = size / 2
  const ring = o.radius_px ? Math.min(Math.max(o.radius_px * 1.15, size * 0.06), size * 0.46) : size * 0.12
  return (
    <div className={`object-crop${large ? ' large' : ''}`}>
      <img src={`${cropUrl(d.stain, o)}${attempt ? `&try=${attempt}` : ''}`} alt={`${o.value} at ${o.tile_id}`} loading="lazy"
        onError={() => { if (attempt < 3) setTimeout(() => setAttempt(attempt + 1), 3000); else setFailed(true) }} />
      <svg viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
        {o.outline ? <>
          <polygon className="mark-halo" points={o.outline.map((p) => p.join(',')).join(' ')} />
          <polygon className="mark" points={o.outline.map((p) => p.join(',')).join(' ')} style={{ stroke: color }} />
        </> : <>
          <circle className="mark-halo" cx={c} cy={c} r={ring} />
          <circle className={`mark${o.radius_px ? '' : ' unknown-size'}`} cx={c} cy={c} r={ring} style={{ stroke: color }} />
        </>}
      </svg>
    </div>
  )
}

/** This object's picture as the example people see for its class (and type) when they label. */
function UseAsExample({ o, stain }: { o: LabelledObject; stain: string }) {
  const [state, setState] = useState<'idle' | 'saving' | 'done'>('idle')
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { setState('idle'); setError(null) }, [o.tile_id, o.x, o.y])
  const use = async () => {
    setState('saving')
    const done = await withError(() => api.post(`/api/labels/classes/${enc(stain)}/examples/from-object`,
      { tile_id: o.tile_id, x: o.x, y: o.y, identity: o.identity, subtype: o.subtype }), setError)
    setState(done ? 'done' : 'idle')
  }
  return (
    <div className="row">
      <button className="btn small" disabled={state !== 'idle'} onClick={use}>{state === 'saving' ? 'Saving…' : `Use as the example picture of “${o.value}”`}</button>
      {state === 'done' && <span className="small status-finished">✓ People labelling now see it (Settings → Label classes)</span>}
      <ErrorNote error={error} />
    </div>
  )
}

function ObjectDetail({ o, d, position, onPrevious, onNext, onClose }: {
  o: LabelledObject; d: SessionObjects; position: string; onPrevious?: () => void; onNext?: () => void; onClose: () => void
}) {
  const place = usePlace()
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft' && onPrevious) onPrevious()
      if (e.key === 'ArrowRight' && onNext) onNext()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onPrevious, onNext])
  const um = d.crop?.pixel_um ?? 0
  return (
    <Modal title={o.value} onClose={onClose} wide>
      <div className="object-detail">
        <ObjectPicture o={o} d={d} large />
        <div className="object-facts">
          <table className="compact"><tbody>
            {o.labels.map((l, i) => (
              <tr key={i}><td><span className="swatch round" style={{ background: classColor(d.groups, l.value) }} /> {l.value}</td>
                <td className="secondary">{who(l.source)}{l.source.startsWith('import: ') ? ' (earlier label set)' : ''}{l.time ? ` · ${timeAgo(l.time)}` : ''}</td></tr>
            ))}
          </tbody></table>
          <AiLine o={o} long />
          <div className="small secondary">{place(o)}{o.disease_group ? ` · ${o.disease_group}` : ''}</div>
          <div className="small muted">
            {o.outline ? `Outline drawn by ${o.outline_by === 'person' ? 'a person' : 'detection'}` : o.radius_px ? 'Ring: the size detection measured' : 'Dashed ring: size not known'}
            {o.radius_px && um ? ` · about ${fmt(2 * o.radius_px * um, 2)} µm across` : ''}</div>
          <Advanced hint="where exactly, whole field">
            <table className="compact"><tbody>
              <tr><td className="secondary">{o.frame === 'core' ? 'Core' : o.frame === 'pilot' ? 'Pilot field' : 'Field'}</td><td><code>{o.tile_id}</code></td></tr>
              <tr><td className="secondary">Position</td><td>{fmt(o.x)}, {fmt(o.y)} px {o.frame === 'field' ? 'in the field picture (with its margin)' : o.frame === 'core' ? 'on the core image' : 'in the pilot picture'}</td></tr>
              {o.core_x != null && <tr><td className="secondary">On the core image</td><td>{fmt(o.core_x)}, {fmt(o.core_y)} px</td></tr>}
            </tbody></table>
            <div className="row">
              {o.core_id && o.core_x != null && <Link className="btn small" to={`/cores/${enc(o.core_id)}?stain=${enc(d.stain)}&x=${o.core_x}&y=${o.core_y}`}>Show on the core</Link>}
              {o.frame === 'field' && <a className="btn small" href={fieldImage(o.tile_id)} target="_blank" rel="noreferrer">Open the whole field picture</a>}
            </div>
            {o.identity && o.crop.state === 'ready' && <UseAsExample o={o} stain={d.stain} />}
          </Advanced>
        </div>
      </div>
      <div className="row">
        <button className="btn small" disabled={!onPrevious} onClick={onPrevious}>‹ Previous</button>
        <span className="small muted">{position}</span>
        <button className="btn small" disabled={!onNext} onClick={onNext}>Next ›</button>
      </div>
    </Modal>
  )
}
