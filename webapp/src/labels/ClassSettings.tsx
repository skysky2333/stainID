/* Settings → Label classes: what people choose from when they label, per stain. The list is always shown; changing it
   (names, keys, descriptions, example pictures, hiding, adding types and classes) is under Advanced. Built-in classes can
   be hidden and renamed but not removed, and keep what the stain's model learns from them (their role). */
import { useEffect, useState } from 'react'
import type { ClassDef, ClassRole, ClassSettings as Settings, KeyMap, SubtypeDef, Vocabulary } from '../api'
import { api, classExample, enc } from '../api'
import { ErrorNote, Loaded, withError } from '../components'
import { useFetch } from '../hooks'
import { Advanced, UploadButton } from '../setup/parts'
import { keyGroups, keyLabel, keySummary } from './classKeys'
import { pickSwatch } from './classes'

const ROLE_TEXT: Record<ClassRole, string> = {
  positive: 'the object the model finds', negative: 'not the object (teaches the model what to leave out)', skip: 'left out of training',
}
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)

export function ClassSettings() {
  const state = useFetch<Settings>('/api/labels/classes')
  const [stain, setStain] = useState('')
  return (
    <section className="card class-settings" id="label-classes">
      <h2>Label classes</h2>
      <p className="muted small">What people choose from when they label, one key each: numbers for the counted types (compact, diffuse… or
        pretangle, mature tangle…), letters for the named look-alikes (CAA, a thread…), Del for “not a plaque”, ` for unsure. The detection
        models learn the class; the types are kept for the per-type measures and a later model.</p>
      <Loaded state={state} label="Loading the classes…">
        {(data) => {
          const stains = Object.keys(data.stains)
          const shown = stains.includes(stain) ? stain : stains[0]
          return <>
            {stains.length > 1 && <div className="segmented">{stains.map((s) => (
              <button key={s} className={`btn small${s === shown ? ' chosen' : ''}`} aria-pressed={s === shown} onClick={() => setStain(s)}>{s}</button>
            ))}</div>}
            {shown && <StainClasses key={shown} vocabulary={data.stains[shown]} keys={data.keys?.[shown] ?? { direct: {} }} builtin={data.builtin[shown]}
              onSaved={state.reload} />}
          </>
        }}
      </Loaded>
    </section>
  )
}

function StainClasses({ vocabulary, keys, builtin, onSaved }: { vocabulary: Vocabulary; keys: KeyMap; builtin: Vocabulary; onSaved: () => void }) {
  return (
    <>
      <ClassList vocabulary={vocabulary} keys={keys} />
      <Advanced title="Change the classes" hint="names, keys, descriptions, example pictures, hide, add types and classes">
        <ClassEditor vocabulary={vocabulary} builtin={builtin} onSaved={onSaved} />
      </Advanced>
    </>
  )
}

/** The keys as people see them on the labelling pages, in their groups, with each one's colour and example. */
function ClassList({ vocabulary, keys }: { vocabulary: Vocabulary; keys: KeyMap }) {
  const hidden = vocabulary.classes.filter((c) => c.hidden).length
  const exampleOf = (identity: string, subtype: string | null) => {
    const c = vocabulary.classes.find((x) => x.id === identity)
    return (subtype ? c?.subtypes.find((s) => s.id === subtype)?.example : '') || (subtype ? '' : c?.example) || ''
  }
  return (
    <div className="class-overview">
      {keyGroups(vocabulary, keys).map((g) => (
        <div key={g.group} className="class-overview-group">
          <div className="small muted">{g.title} <span className="kbd-text">{g.keys}</span></div>
          {g.items.map((item) => {
            const example = exampleOf(item.identity, item.subtype)
            return (
              <div key={item.key} className="class-overview-row">
                <span className="kbd">{item.label}</span><span className="swatch" style={pickSwatch(vocabulary, item.identity, item.subtype)} />
                <span><b>{item.name}</b>{item.description && <span className="small secondary class-description">{item.description}</span>}</span>
                {example && <span className="class-examples"><img src={classExample(vocabulary.stain, example)} alt={`Example: ${item.name}`} title={item.name} /></span>}
              </div>
            )
          })}
        </div>
      ))}
      <p className="muted small" style={{ margin: 0 }}>{keySummary(vocabulary, keys)}.
        {hidden > 0 && ` ${hidden} hidden class${hidden === 1 ? '' : 'es'} not shown to people.`}</p>
    </div>
  )
}

function ClassEditor({ vocabulary, builtin, onSaved }: { vocabulary: Vocabulary; builtin: Vocabulary; onSaved: () => void }) {
  const [classes, setClasses] = useState<ClassDef[]>(() => structuredClone(vocabulary.classes))
  const [error, setError] = useState<string | null>(null)
  const [state, setState] = useState<'idle' | 'saving' | 'saved'>('idle')
  useEffect(() => { setClasses(structuredClone(vocabulary.classes)) }, [vocabulary])
  const dirty = !same(classes, vocabulary.classes)
  const atDefaults = same(vocabulary.classes.map(strip), builtin.classes.map(strip))
  const change = (next: ClassDef[]) => { setState('idle'); setClasses(next) }
  const setClass = (i: number, c: ClassDef) => change(classes.map((x, j) => (j === i ? c : x)))
  const move = (i: number, by: number) => { const next = [...classes]; [next[i], next[i + by]] = [next[i + by], next[i]]; change(next) }
  const save = async (body: { classes: ClassDef[] | null }) => {
    setState('saving')
    const done = await withError(() => api.put(`/api/labels/classes/${enc(vocabulary.stain)}`, body), setError)
    setState(done ? 'saved' : 'idle')
    if (done) onSaved()
  }
  return (
    <div className="class-editor">
      {classes.map((c, i) => (
        <ClassRow key={c.id || `new-${i}`} stain={vocabulary.stain} c={c} first={i === 0} last={i === classes.length - 1}
          onChange={(next) => setClass(i, next)} onMove={(by) => move(i, by)} onRemove={() => change(classes.filter((_, j) => j !== i))} />
      ))}
      <div className="row">
        <button className="btn small" onClick={() => change([...classes, { id: '', name: '', role: 'negative', description: '', key: '', example: '', hidden: false, subtypes: [] }])}>
          Add a class</button>
        <span className="spacer" />
      </div>
      <div className="row">
        <button className="btn primary" disabled={!dirty || state === 'saving'} onClick={() => save({ classes })}>
          {state === 'saving' ? 'Saving…' : 'Save classes'}</button>
        {dirty && <button className="btn" onClick={() => setClasses(structuredClone(vocabulary.classes))}>Undo changes</button>}
        {!atDefaults && !dirty && <button className="btn" onClick={() => save({ classes: null })}>Back to the built-in classes</button>}
        {state === 'saved' && !dirty && <span className="small status-finished">✓ Saved to stainid.yaml. Open labelling pages use them when reopened.</span>}
      </div>
      <p className="muted small" style={{ margin: 0 }}>Labels people already made keep their class: renaming a class or a type changes only its name; hidden ones are
        no longer offered but their labels stay. Keys: a counted type takes a number (built in 1–4, yours 5–9), anything else a letter (built in
        Q W E R, yours T Y U …); left empty, one is given. Del, `, 0, B, S, Z, V, A and O are used by the labelling pages.</p>
      <ErrorNote error={error} />
    </div>
  )
}

const strip = (c: ClassDef) => ({ ...c, builtin: undefined, subtypes: c.subtypes.map((s) => ({ ...s, builtin: undefined })) })

function ClassRow({ stain, c, first, last, onChange, onMove, onRemove }: {
  stain: string; c: ClassDef; first: boolean; last: boolean; onChange: (c: ClassDef) => void; onMove: (by: number) => void; onRemove: () => void
}) {
  const setType = (i: number, s: SubtypeDef) => onChange({ ...c, subtypes: c.subtypes.map((x, j) => (j === i ? s : x)) })
  const moveType = (i: number, by: number) => { const next = [...c.subtypes]; [next[i], next[i + by]] = [next[i + by], next[i]]; onChange({ ...c, subtypes: next }) }
  return (
    <div className={`class-edit${c.hidden ? ' is-hidden' : ''}`}>
      <div className="row class-edit-head">
        {c.key === 'delete' || c.key === '`' || c.role === 'skip' ? <span className="kbd" title="Fixed">{keyLabel(c.key || '`')}</span>
          : <input type="text" value={c.key} maxLength={1} style={{ width: 38 }} aria-label="Key" title="Key (a number for the counted object, else a letter)"
            onChange={(e) => onChange({ ...c, key: e.target.value.toLowerCase() })} />}
        <input type="text" value={c.name} placeholder="Name of the class" style={{ flex: '1 1 160px' }} aria-label="Name" onChange={(e) => onChange({ ...c, name: e.target.value })} />
        {c.builtin ? <span className="muted small">{ROLE_TEXT[c.role]}</span> : (
          <select value={c.role} aria-label="What the model learns from it" onChange={(e) => onChange({ ...c, role: e.target.value as ClassRole })}>
            {(Object.keys(ROLE_TEXT) as ClassRole[]).map((r) => <option key={r} value={r}>{ROLE_TEXT[r]}</option>)}
          </select>
        )}
        <label className="layer-toggle"><input type="checkbox" checked={c.hidden} onChange={(e) => onChange({ ...c, hidden: e.target.checked })} />hidden</label>
        <button className="btn small" disabled={first} aria-label="Move up" title="Move up" onClick={() => onMove(-1)}>↑</button>
        <button className="btn small" disabled={last} aria-label="Move down" title="Move down" onClick={() => onMove(1)}>↓</button>
        {!c.builtin && <button className="link-btn small" onClick={onRemove}>Remove</button>}
      </div>
      <textarea rows={1} value={c.description} placeholder="What it looks like, in plain words" aria-label={`Description of ${c.name}`}
        onChange={(e) => onChange({ ...c, description: e.target.value })} />
      <ExampleField stain={stain} file={c.example} name={c.name} onChange={(example) => onChange({ ...c, example })} />
      <div className="type-edits">
        {c.subtypes.map((s, i) => (
          <div key={s.id || `new-${i}`} className={`row type-edit${s.hidden ? ' is-hidden' : ''}`}>
            <input type="text" value={s.name} placeholder="Name of the type" style={{ flex: '1 1 140px' }} aria-label="Type name" onChange={(e) => setType(i, { ...s, name: e.target.value })} />
            <input type="text" value={s.key} maxLength={1} style={{ width: 38 }} aria-label="Key of the type" title="Key of the type (a number for a counted type)"
              onChange={(e) => setType(i, { ...s, key: e.target.value.toLowerCase() })} />
            <input type="text" value={s.description} placeholder="What it looks like" style={{ flex: '2 1 200px' }} aria-label={`Description of ${s.name}`}
              onChange={(e) => setType(i, { ...s, description: e.target.value })} />
            <label className="layer-toggle"><input type="checkbox" checked={s.hidden} onChange={(e) => setType(i, { ...s, hidden: e.target.checked })} />hidden</label>
            <ExampleField stain={stain} file={s.example} name={s.name} compact onChange={(example) => setType(i, { ...s, example })} />
            <button className="btn small" disabled={i === 0} aria-label="Move up" onClick={() => moveType(i, -1)}>↑</button>
            <button className="btn small" disabled={i === c.subtypes.length - 1} aria-label="Move down" onClick={() => moveType(i, 1)}>↓</button>
            {!s.builtin && <button className="link-btn small" onClick={() => onChange({ ...c, subtypes: c.subtypes.filter((_, j) => j !== i) })}>Remove</button>}
          </div>
        ))}
        <button className="link-btn small" onClick={() => onChange({ ...c, subtypes: [...c.subtypes, { id: '', name: '', description: '', key: '', example: '', hidden: false }] })}>
          + Add a type of {c.name || 'this class'}</button>
      </div>
    </div>
  )
}

/** The example picture of a class or type: shown small, replaced by uploading a picture, or removed. */
function ExampleField({ stain, file, name, compact = false, onChange }: { stain: string; file: string; name: string; compact?: boolean; onChange: (file: string) => void }) {
  const upload = async (picture: File) => {
    const made = await api.upload<{ file: string }>(`/api/labels/classes/${enc(stain)}/examples`, picture)
    onChange(made.file)
    return compact ? '' : 'Added — save to keep it'
  }
  return (
    <span className="example-field">
      {file && <img src={classExample(stain, file)} alt={`Example: ${name}`} />}
      <UploadButton label={file ? 'Other picture…' : compact ? 'Picture…' : 'Example picture…'} accept="image/*" onFile={upload} />
      {file && <button className="link-btn small" onClick={() => onChange('')}>Remove picture</button>}
    </span>
  )
}
