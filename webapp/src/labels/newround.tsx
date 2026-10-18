/* Making an annotation round (the Labels page): "Fix the field" (objects) or "Mark bad areas" (the brush), where its objects come from (every ready core, some
   TMAs, or fields chosen on the cores, each core showing whether it is ready) and how many. Everything else (how
   generous the candidates are, how many people must decide each object, the session ID, a note, types) is under
   Advanced, with the older kinds of session (whole-field review and quick check of the label pool). */
import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { AreasOverview, RoundScope, SessionInfo, Vocabulary } from '../api'
import { api } from '../api'
import { ErrorNote, Loaded, Modal, withError } from '../components'
import { useFetch } from '../hooks'
import { useProject } from '../project'
import { Advanced } from '../setup/parts'
import { FieldPicker } from './fields'
import { AREAS_TEXT, AREAS_TITLE } from './markareas'
import { ROUND_TEXT, ROUND_TITLE } from './rounds'
import { byKey } from './classKeys'

type Where = 'all' | 'tmas' | 'pick'
/** What the project's pictures and their groups are called: cores on TMAs, or slides / images in batches. */
const WORDS = { tma: { pictures: 'cores', groups: 'TMAs', group: 'TMA' }, slides: { pictures: 'slides', groups: 'batches', group: 'batch' },
  images: { pictures: 'images', groups: 'batches', group: 'batch' } } as const
const LEVEL_TEXT: Record<string, string> = {
  all: 'Every candidate detection proposed (most generous)', '0.5': 'Proposed at half the stain threshold or stronger',
  '0.75': 'At three quarters of the stain threshold or stronger', '1': 'At the stain threshold or stronger (as detection counts)',
  '1.5': 'Only clearly stained ones (1.5 × the threshold)',
}

export function NewRoundSession({ stains, onClose, onCreated, onOlder }: {
  stains: string[]; onClose: () => void; onCreated: (s: SessionInfo) => void; onOlder: () => void
}) {
  const { tmaName, info } = useProject()
  const words = WORDS[info?.kind ?? 'tma'] ?? WORDS.tma
  const [stain, setStain] = useState(stains[0] ?? 'NeuN')
  const [round, setRound] = useState<2 | 'areas'>(2)  // the field tool is the default: every object of a field, corrected in place
  const [order, setOrder] = useState<'problems' | 'random'>('problems')  // "Mark bad areas": likely problems first
  const [where, setWhere] = useState<Where>('all')
  const [tmas, setTmas] = useState<string[]>([])
  const [chosen, setChosen] = useState<string[]>([])
  const [fieldCount, setFieldCount] = useState(-1)  // -1: every remaining field
  const [spread, setSpread] = useState(false)  // Advanced: spread the fields over TMAs and groups instead of a plain random draw
  const [seed, setSeed] = useState('')
  const [title, setTitle] = useState('')
  const [open, setOpen] = useState(true)
  const [level, setLevel] = useState('all')
  const [needed, setNeeded] = useState(1)
  const [name, setName] = useState('')
  const [instructions, setInstructions] = useState('')
  const [subtypes, setSubtypes] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const scope = useFetch<RoundScope>(`/api/labels/rounds/scope?stain=${encodeURIComponent(stain)}`)
  const defaults = useFetch<{ vocabularies: Record<string, Vocabulary> }>('/api/labels/defaults')
  const areasNow = useFetch<AreasOverview>(round === 'areas' ? '/api/labels/areas' : null)
  const areas = round === 'areas'
  const day = new Date().toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
  const placeholder = areas ? `${AREAS_TITLE} – ${day}` : `${stain} · ${ROUND_TITLE[round]} – ${day}`
  const nothing = (where === 'pick' && !chosen.length && !areas) || (where === 'tmas' && !tmas.length)
  const create = async () => {
    setBusy(true)
    const body = areas
      ? { stain: 'all', kind: 'areas', title: title.trim() || placeholder, name, open, instructions, n_fields: fieldCount, order,
        seed: seed.trim() === '' ? null : Number(seed), tmas: where === 'tmas' ? tmas : null }
      : { stain, round, title: title.trim() || placeholder, name, open, instructions, subtypes, labels_needed: needed,
        level: level === 'all' ? null : Number(level), n_fields: fieldCount, spread, seed: seed.trim() === '' ? null : Number(seed),
        tmas: where === 'tmas' ? tmas : null, tile_ids: where === 'pick' ? chosen : null }
    const made = await withError(() => api.post<SessionInfo>('/api/labels/sessions', body), setError)
    setBusy(false)
    if (made) { onCreated(made); onClose() }
  }
  const vocabulary = defaults.data?.vocabularies[stain]
  const typed = vocabulary?.classes.filter((c) => !c.hidden && c.subtypes.some((s) => !s.hidden)) ?? []
  return (
    <Modal title="New labelling session" onClose={onClose} wide>
      <div className="new-session">
        {!areas && <fieldset className="field"><legend>Stain</legend>
          <div className="segmented">{stains.map((s) => (
            <button key={s} type="button" className={`btn${stain === s ? ' chosen' : ''}`} aria-pressed={stain === s}
              onClick={() => { setStain(s); setChosen([]); setTmas([]) }}>{s}</button>
          ))}</div>
        </fieldset>}
        <fieldset className="field"><legend>What should people do?</legend>
          <div className="choice-grid">
            <label className={`choice-card${round === 2 ? ' chosen' : ''}`}>
              <input type="radio" name="round" checked={round === 2} onChange={() => setRound(2)} />
              <span><b>{ROUND_TITLE[2]} (start here)</b><span className="small secondary">{ROUND_TEXT[2]}</span></span>
            </label>
            <label className={`choice-card${areas ? ' chosen' : ''}`}>
              <input type="radio" name="round" checked={areas} onChange={() => { setRound('areas'); if (where === 'pick') setWhere('all') }} />
              <span><b>{AREAS_TITLE}</b><span className="small secondary">{AREAS_TEXT}</span></span>
            </label>
          </div>
          <p className="small muted" style={{ margin: '6px 0 0' }}>{areas
            ? 'Every stain of a core in one go. A sample is enough: the areas people mark teach and test the bad-area detection; the analysis keeps detection’s own areas. About 1 core in 10 also goes to a second person, to see how well people agree.'
            : 'People work one field at a time: a field someone is on, or has finished, is not handed to anyone else. Every session’s answers on a candidate count together, whoever gave them.'}</p>
        </fieldset>
        {areas ? <AreasScope scope={scope.data} overview={areasNow} where={where} setWhere={setWhere} tmas={tmas} setTmas={setTmas}
          count={fieldCount} setCount={setFieldCount} order={order} setOrder={setOrder} words={words} tmaName={info?.kind === 'tma' || !info?.kind ? tmaName : (t) => t} /> : (
        <Loaded state={scope} label="Looking at which cores are ready…">
          {(s) => <>
            <fieldset className="field"><legend>Which {words.pictures}?</legend>
              <p className={`small ${s.cores_ready ? 'secondary' : 'warning-text'}`} style={{ margin: '0 0 8px' }}>
                {s.cores_ready ? <>{s.cores_ready.toLocaleString()} of {s.cores.toLocaleString()} {words.pictures} have detection’s candidates ({s.fields_ready.toLocaleString()} fields).</>
                  : <>No {stain} {words.pictures.slice(0, -1)} has detection’s candidates yet: run the v2 detection first. Until then the older kinds of session (under Advanced) use the current label pool.</>}
                {s.from_pool && <span className="muted"> These candidates were made from the label pool for trying the rounds out.</span>}</p>
              <div className="segmented">
                <button type="button" className={`btn${where === 'all' ? ' chosen' : ''}`} onClick={() => setWhere('all')}>Every ready {words.pictures.slice(0, -1)}</button>
                <button type="button" className={`btn${where === 'tmas' ? ' chosen' : ''}`} onClick={() => setWhere('tmas')}>Some {words.groups}</button>
                <button type="button" className={`btn${where === 'pick' ? ' chosen' : ''}`} onClick={() => setWhere('pick')}>Choose on the {words.pictures}</button>
              </div>
              {where === 'tmas' && <div className="tma-chips">{s.tmas.map((t) => (
                <label key={t.tma} className={`tma-chip${t.cores_ready ? '' : ' is-blocked'}`} title={`${t.fields_ready} of ${t.fields} fields ready`}>
                  <input type="checkbox" disabled={!t.cores_ready} checked={tmas.includes(t.tma)}
                    onChange={() => setTmas(tmas.includes(t.tma) ? tmas.filter((x) => x !== t.tma) : [...tmas, t.tma])} />
                  <b>{info?.kind === 'tma' || !info?.kind ? tmaName(t.tma) : t.tma}</b> <span className="muted small">{t.cores_ready} of {t.cores} {words.pictures} ready</span>
                </label>
              ))}</div>}
              {where === 'pick' && <FieldPicker stain={stain} chosen={chosen} onChange={setChosen} source="candidates" />}
            </fieldset>
            {where !== 'pick' ? (() => {
              const left = where === 'tmas' ? s.tmas.filter((t) => tmas.includes(t.tma)).reduce((n, t) => n + (t.fields_remaining ?? t.fields_ready), 0)
                : s.fields_remaining ?? s.fields_ready
              return (
                <fieldset className="field"><legend>How many fields</legend>
                  <div className="segmented">
                    <button type="button" className={`btn${fieldCount < 0 ? ' chosen' : ''}`} onClick={() => setFieldCount(-1)}>All remaining fields ({left.toLocaleString()})</button>
                    <button type="button" className={`btn${fieldCount > 0 ? ' chosen' : ''}`} onClick={() => setFieldCount(fieldCount > 0 ? fieldCount : Math.min(10, Math.max(1, left)))}>A random sample</button>
                  </div>
                  {fieldCount > 0 && <label className="inline-number small" style={{ marginTop: 6 }}>
                    <input type="number" min={1} value={fieldCount} style={{ width: 90 }} onChange={(e) => setFieldCount(Math.max(1, Number(e.target.value) || 1))} />
                    {' '}fields drawn at random from the {left.toLocaleString()} remaining</label>}
                  <p className="small muted" style={{ margin: '6px 0 0' }}>Remaining: ready fields nobody has finished that are not in another open session.
                    {fieldCount > left && left > 0 ? <b className="warning-text"> Only {left.toLocaleString()} remain, so all of them are taken.</b> : null}</p>
                </fieldset>
              )
            })() : null}
          </>}
        </Loaded>)}
        <label className="field">Title people see<input type="text" value={title} placeholder={placeholder} onChange={(e) => setTitle(e.target.value)} /></label>
        <label className="layer-toggle"><input type="checkbox" checked={open} onChange={(e) => setOpen(e.target.checked)} />Open to labellers right away</label>
        {areas ? <Advanced hint="session ID, random seed, note">
          <div className="row" style={{ alignItems: 'flex-end' }}>
            <label className="field">Session ID (in links and files)<input type="text" value={name} placeholder="made from the title"
              onChange={(e) => setName(e.target.value.replace(/[^A-Za-z0-9_-]/g, '_'))} /></label>
            <label className="field">Random seed<input type="number" value={seed} placeholder="new each time" style={{ width: 130 }}
              onChange={(e) => setSeed(e.target.value)} /></label>
          </div>
          <label className="field">Note for labellers (optional)<textarea rows={2} value={instructions} placeholder="e.g. Pay attention to the edges of the core."
            onChange={(e) => setInstructions(e.target.value)} /></label>
        </Advanced> : <Advanced hint="how generous, people per field, session ID, note, types, older kinds of session">
          <label className="field">How generous (which candidates are shown)
            <select value={level} onChange={(e) => setLevel(e.target.value)}>
              {Object.keys(scope.data?.levels ?? LEVEL_TEXT).map((k) => <option key={k} value={k}>{LEVEL_TEXT[k] ?? k}</option>)}</select></label>
          <div className="row" style={{ alignItems: 'flex-end' }}>
            <label className="field">People who must finish each field<input type="number" min={1} max={9} value={needed} style={{ width: 80 }}
              onChange={(e) => setNeeded(Math.max(1, Number(e.target.value)))} /></label>
            <label className="field">Session ID (in links and files)<input type="text" value={name} placeholder="made from the title"
              onChange={(e) => setName(e.target.value.replace(/[^A-Za-z0-9_-]/g, '_'))} /></label>
          </div>
          {round === 2 && where !== 'pick' && <div className="row" style={{ alignItems: 'flex-end' }}>
            <label className="layer-toggle"><input type="checkbox" checked={spread} onChange={(e) => setSpread(e.target.checked)} />
              Spread the fields over {words.groups} and groups (instead of a plain random draw)</label>
            <label className="field">Random seed<input type="number" value={seed} placeholder="new each time" style={{ width: 130 }}
              onChange={(e) => setSeed(e.target.value)} /></label>
          </div>}
          <label className="field">Note for labellers (optional)<textarea rows={2} value={instructions} placeholder="e.g. Count only neurons with a visible nucleus."
            onChange={(e) => setInstructions(e.target.value)} /></label>
          {vocabulary && <p className="muted small" style={{ margin: 0 }}>Classes: {byKey(vocabulary, { direct: {} }).map((c) => c.name).join(' · ')}.
            {' '}<Link to="/settings#label-classes">Change the classes</Link></p>}
          {typed.length > 0 && <label className="layer-toggle"><input type="checkbox" checked={subtypes} onChange={(e) => setSubtypes(e.target.checked)} />
            Also ask for the type ({typed.map((c) => c.name).join(', ')})</label>}
          <p className="small secondary" style={{ marginBottom: 0 }}>Older kinds of session — the whole-field review and quick check on the current label
            pool: <button className="link-btn" onClick={onOlder}>make one of those instead…</button></p>
        </Advanced>}
        <ErrorNote error={error ?? scope.error} />
        <div className="row"><span className="spacer" />
          {nothing && <span className="small muted">{where === 'pick' ? `Click squares on the ${words.pictures} above to choose fields.` : `Tick at least one ${words.group}.`}</span>}
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn primary" disabled={busy || nothing || (!areas && !scope.data?.cores_ready)} onClick={create}>{busy ? 'Creating…' : 'Create session'}</button>
        </div>
      </div>
    </Modal>
  )
}

/** "Mark bad areas": which cores (every one, or some TMAs), how many (every one not checked yet, or a sample) and in
 *  which order (likely problems first, or at random). */
function AreasScope({ scope, overview, where, setWhere, tmas, setTmas, count, setCount, order, setOrder, words, tmaName }: {
  scope: RoundScope | null; overview: { data: AreasOverview | null; error: string | null }
  where: Where; setWhere: (w: Where) => void; tmas: string[]; setTmas: (t: string[]) => void
  count: number; setCount: (n: number) => void; order: 'problems' | 'random'; setOrder: (o: 'problems' | 'random') => void
  words: (typeof WORDS)[keyof typeof WORDS]; tmaName: (tma: string) => string
}) {
  const o = overview.data
  const left = o ? o.cores - o.cores_checked : null
  return <>
    <fieldset className="field"><legend>Which {words.pictures}?</legend>
      {o && <p className="small secondary" style={{ margin: '0 0 8px' }}>{o.cores_checked.toLocaleString()} of {o.cores.toLocaleString()} {words.pictures} have
        been checked for bad areas so far.</p>}
      <div className="segmented">
        <button type="button" className={`btn${where !== 'tmas' ? ' chosen' : ''}`} onClick={() => setWhere('all')}>Every {words.pictures.slice(0, -1)}</button>
        <button type="button" className={`btn${where === 'tmas' ? ' chosen' : ''}`} onClick={() => setWhere('tmas')}>Some {words.groups}</button>
      </div>
      {where === 'tmas' && <div className="tma-chips">{(scope?.tmas ?? []).map((t) => (
        <label key={t.tma} className="tma-chip">
          <input type="checkbox" checked={tmas.includes(t.tma)} onChange={() => setTmas(tmas.includes(t.tma) ? tmas.filter((x) => x !== t.tma) : [...tmas, t.tma])} />
          <b>{tmaName(t.tma)}</b> <span className="muted small">{t.cores} {words.pictures}</span>
        </label>
      ))}</div>}
    </fieldset>
    <fieldset className="field"><legend>How many {words.pictures}</legend>
      <div className="segmented">
        <button type="button" className={`btn${count < 0 ? ' chosen' : ''}`} onClick={() => setCount(-1)}>All not checked yet{left != null && where !== 'tmas' ? ` (${left.toLocaleString()})` : ''}</button>
        <button type="button" className={`btn${count > 0 ? ' chosen' : ''}`} onClick={() => setCount(count > 0 ? count : 30)}>A sample</button>
      </div>
      {count > 0 && <label className="inline-number small" style={{ marginTop: 6 }}>
        <input type="number" min={1} value={count} style={{ width: 90 }} onChange={(e) => setCount(Math.max(1, Number(e.target.value) || 1))} />
        {' '}{words.pictures}, taken in the order below</label>}
      <p className="small muted" style={{ margin: '6px 0 0' }}>People stop whenever they like: the session is a sample, it does not need every {words.pictures.slice(0, -1)} checked.
        {' '}{words.pictures[0].toUpperCase() + words.pictures.slice(1)} someone checked already, or in another open session, are left out.</p>
    </fieldset>
    <fieldset className="field"><legend>Order</legend>
      <div className="segmented">
        <button type="button" className={`btn${order === 'problems' ? ' chosen' : ''}`} onClick={() => setOrder('problems')}>Likely problems first</button>
        <button type="button" className={`btn${order === 'random' ? ' chosen' : ''}`} onClick={() => setOrder('random')}>Random</button>
      </div>
      <p className="small muted" style={{ margin: '6px 0 0' }}>{order === 'problems'
        ? `${words.pictures[0].toUpperCase() + words.pictures.slice(1)} where detection found much to leave out, or little tissue, come first; then the rest at random.`
        : `${words.pictures[0].toUpperCase() + words.pictures.slice(1)} in a random order: a fair sample for testing the bad-area detection.`}</p>
    </fieldset>
    <ErrorNote error={overview.error} />
  </>
}
