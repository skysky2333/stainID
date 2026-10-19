/* The two tutorials (/tutorial), offered on everyone's home and on Help (`TutorialCards`): "Fix the field" (about 3–5
   minutes) on the practice field, and `?kind=areas`, "Mark bad areas", on the practice core (labels/markareas.tsx): the
   one tutorial of the server (`stainid tutorial build`, a public dataset's field and core, the same for every project),
   with its credit line. Nothing is saved: the practice sessions (`practice-2-<stain>`, `practice-areas`)
   keep nothing on the server, whatever is clicked. A coach asks for one thing at a time and waits until the person has
   done it. Skip at any time (back to `?next=`, else home). Finishing or skipping is kept with the person's account
   (`/api/session/tutorials`); someone trying it without an account keeps it in this browser. The admin can open either from the Labels page (`?preview=1`): the same pages, nothing remembered. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import type { MySet, SessionDetail } from '../api'
import { api, appBase, projectPath } from '../api'
import { Empty, ErrorNote, Loaded, Loading, PageHead } from '../components'
import { useFetch } from '../hooks'
import { FixField } from './fixfield'
import type { AreaEvent } from './markareas'
import { MarkAreas, PRACTICE_AREAS } from './markareas'
import type { RoundEvent } from './rounds'
import { useSession } from '../session'

/** `/api/tutorial`: the one tutorial of the server (`stainid.tutorial`, the same for every project): its field, the
 *  credit line of its images and the hint of the "Change the type" step; null before `stainid tutorial build`. */
interface TutorialInfo {
  done: boolean
  practice: { stain: string; tile_id: string; round2: string; credit?: string; hint?: { text?: string; key?: string } } | null
}
export type TutorialKind = 'objects' | 'areas'
interface Done { objects: boolean; areas: boolean; kept: boolean }
/** One step: done once the page reports one of `done` (`times` of them: "reject a few"). */
interface Step { title: string; text: string; done: RoundEvent[]; times?: number }

/** The steps, building up: first the acceptance threshold (step 1 of every field: move it, watch objects switch, Start
 *  editing), a clean look (Space), then the object changes — reject what detection got wrong (Del), bring types back
 *  with a number, change a type (on a stain with types), unsure (`), add a missed object, split (S), merge (M) — then
 *  the outlines: Tab, precise editing (click / Shift+click a spot, then drag the dots), then the brush (B), saving. `keys`: the practice
 *  stain's key line (`Vocabulary.keymap` summary); `typed`: its objects have several types to choose from. */
function stepsFor(typed: boolean, counted: string, hint = ''): Step[] {
  const steps: Step[] = [
    { title: 'Step 1: the acceptance threshold', text: 'Every field starts with the slider at the top of the panel, at detection’s own threshold (marked on the track). Drag it (or press [ and ]): objects scoring at or above it turn solid, those below dotted grey. It only sets where this field starts; the analysis keeps detection’s threshold.', done: ['threshold'] },
    { title: 'Start editing', text: 'When the field starts about right, press “Start editing” (or Enter). The slider locks and shows a lock; your first edit would lock it too. Only “Reset field” unlocks it again, and that discards the field’s unsaved changes.', done: ['start'] },
    { title: 'A clean look', text: 'Press Space to hide everything drawn over the picture and see the tissue alone. Press Space again to bring it back.', done: ['overlays'] },
    { title: 'Point at an object', text: 'Move the mouse over an outlined object. It lights up, its name shows beside it, and what detection said about it shows on the right. Every key acts on the object under the mouse.', done: ['hover'] },
    { title: 'Reject what detection got wrong', text: 'Find a few objects detection got wrong, point at each and press Del. They turn dotted grey and stay visible. Some objects are already dotted grey: detection rejected those itself, and they look just the same.', done: ['remove'], times: 2 },
    { title: 'Bring the type back', text: `Point at a dotted grey object that is a real one and press ${counted}: it becomes solid in that type’s colour. Each type has its own colour (see the panel on the right).`, done: ['class'] },
    { title: 'Change the type', text: `Point at a solid object and press another number: it changes to that type and colour. ${hint ? `${hint} ` : ''}Things that look like one but are not have letters (Q, W …): solid too, but not counted.`, done: ['subtype'] },
    { title: 'Not sure?', text: 'Point at an object you cannot decide and press ` (the key left of 1; 0 works too). It turns dotted amber with a “?” and is left out of training and evaluation.', done: ['unsure'] },
    { title: 'Undo', text: 'Pressed the wrong key? Press Z to undo your last change. Z takes back changes you have not saved yet.', done: ['undo'] },
    { title: 'Add a missed object', text: `Click on tissue where detection missed an object: its outline appears as a new candidate (dotted grey with a diamond). Then press ${counted} to make it solid. (Where one-click outlines are not available, the click starts an outline by hand: click its corners, then the first one.)`, done: ['added'] },
    { title: 'Split an object', text: 'Two objects outlined as one? Press S for split, then click the object: a split into two shows. Click elsewhere on it to move the split, press Enter to keep it (Esc drops it). You can also drag a line right across objects to cut each one along it. The pieces keep the type; the keys retype them.', done: ['split'] },
    { title: 'Merge objects', text: 'One object outlined in pieces? Press M for merge, click the pieces (or drag over them) — the panel counts them — and press Enter. They become one object with the type of the largest. Press M or Esc to leave.', done: ['merge'] },
    { title: 'Tab switches the outline', text: 'Point at an object and press Tab: its outline switches from the stain outline to Segment Anything’s small one, then its medium one, then back (the panel says which is shown), and the one you leave it on is saved with the field. (With the brush on, Tab sizes the brush instead.)', done: ['mask'] },
    { title: 'Precise editing', text: 'For an outline that is not right: click the object. It is zoomed in on, and everything else is locked until you leave. (To change its type you only point at it: no click needed.)', done: ['precise-in'] },
    { title: 'Add or leave out a spot', text: 'Click a spot of the object to add it, Shift+click one to leave it out: the outline is drawn again around what you mean. Z takes one step back.', done: ['precise-spot'] },
    { title: 'Leave precise editing', text: 'Click outside the faint frame around the object (or press Enter) to keep the outline: the view goes back to where you were (Esc would cancel the edit).', done: ['precise-out'] },
    { title: 'Drag the dots', text: 'Click an object again for precise editing and drag one of the white dots on its outline to where the edge is (on a server without one-click outlines, dragging the dots is the way). Click outside the faint frame (or Enter) to keep it.', done: ['precise-drag'] },
    { title: 'Mark a bad area', text: 'Hatched areas are what the analysis leaves out: folds, artifacts, empty glass, found by detection. Press B for the brush: 1–4 choose the type (fold, artifact, no tissue, out of focus), and that type’s area shows as one region — detection’s with everyone’s edits. Drag to add to it (a stroke touching a detected fold extends that fold); Shift-drag takes a part out, detection’s too. Tab makes the brush smaller, Shift+Tab bigger; the wheel still zooms. Your areas teach and test the bad-area detection; the analysis keeps detection’s own. Press B again when done.', done: ['exclude'] },
    { title: 'Save your work', text: 'Nothing you do is saved yet: the panel says how many unsaved changes you have, and a small orange dot marks each object you changed. Press Save (or Ctrl+S) to keep them — you stay on this field. (In this practice, Save only pretends.)', done: ['save'] },
    { title: 'Save and continue', text: 'When the field looks right, press “Save and continue”. Everything you left as it is counts as checked by you, the field is done, nobody else is asked to do it again, and your next field opens.', done: ['finish'] },
  ]
  return typed ? steps : steps.filter((s) => s.title !== 'Change the type')
}

const guestKey = (kind: TutorialKind) => `stainid.tutorial.${kind}`
function guestDone(kind: TutorialKind) {
  try { return localStorage.getItem(guestKey(kind)) === '1' } catch { return false }
}

/** Which tutorials this person went through or skipped: kept with their account, a guest's in this browser. */
export function useTutorialsDone() {
  const server = useFetch<Done>('/api/session/tutorials')
  const kept = !!server.data?.kept
  return { objects: kept ? server.data!.objects : guestDone('objects'), areas: kept ? server.data!.areas : guestDone('areas'), loaded: !!server.data || !!server.error }
}

/** Remember that this person finished or skipped a tutorial (`done` false: offer it again). */
export async function rememberTutorial(kind: TutorialKind, done = true) {
  try { if (done) localStorage.setItem(guestKey(kind), '1'); else localStorage.removeItem(guestKey(kind)) } catch { /* a private window */ }
  await api.post('/api/session/tutorials', { kind, done })
}

const TUTORIALS: { kind: TutorialKind; title: string; text: string; minutes: string }[] = [
  { kind: 'objects', title: 'Fix the field', minutes: '3–5 minutes', text: 'The main task: correct what detection found on a field — reject, change types, fix outlines, split, merge, add what it missed.' },
  { kind: 'areas', title: 'Mark bad areas', minutes: '2 minutes', text: 'The brush: paint folds, artifacts, missing tissue and blur that should not be counted, core by core with every stain.' },
]

/** The address of a tutorial: in the page's project, else in the project of the person's first set, else the project
 *  the server opened last (every project practises on the same tutorial, `stainid.tutorial`). */
export function tutorialPath(kind: TutorialKind, sets?: MySet[] | null, next?: string) {
  const query = [kind === 'areas' ? 'kind=areas' : '', next ? `next=${encodeURIComponent(next)}` : ''].filter(Boolean).join('&')
  const page = `/tutorial${query ? `?${query}` : ''}`
  if (appBase || !sets?.length) return { to: page, href: `${appBase}${page}` }
  return { to: null, href: projectPath(sets[0].project, page) }
}

/** Both tutorials, offered on everyone's home and on Help: what each is for, and whether this person has done it. */
export function TutorialCards({ sets, compact = false }: { sets?: MySet[] | null; compact?: boolean }) {
  const done = useTutorialsDone()
  return (
    <div className={`grid grid-2 tutorial-cards${compact ? ' compact' : ''}`}>
      {TUTORIALS.map((t) => {
        const { to, href } = tutorialPath(t.kind, sets)
        const label = done[t.kind] ? 'See it again' : 'Start the tutorial'
        return (
          <div key={t.kind} className="card tutorial-card">
            <div className="row"><b>Tutorial: {t.title}</b><span className="spacer" />
              {done.loaded && (done[t.kind] ? <span className="status-finished small">✓ done</span> : <span className="muted small">{t.minutes}</span>)}</div>
            {!compact && <p className="small secondary" style={{ margin: '4px 0 8px' }}>{t.text} Nothing is saved.</p>}
            {to ? <Link className={`btn${done[t.kind] ? '' : ' primary'}`} to={to}>{label}</Link>
              : <a className={`btn${done[t.kind] ? '' : ' primary'}`} href={href}>{label}</a>}
          </div>
        )
      })}
    </div>
  )
}

/** Where to go back to after a tutorial: `?next=` when it is an address of this app, else home. */
function nextOf(params: URLSearchParams) {
  const next = params.get('next') ?? ''
  return next.startsWith('/') && !next.startsWith('//') ? next : '/'
}

export function Tutorial() {
  const { session } = useSession()
  const { search } = useLocation()
  const params = new URLSearchParams(search)
  const preview = params.get('preview') === '1' && session.admin
  const next = nextOf(params)
  const areas = params.get('kind') === 'areas'
  const info = useFetch<TutorialInfo>(!areas ? '/api/tutorial' : null)  // public: no account needed, nothing is kept
  if (areas) return <AreasCoached preview={preview} next={next} />
  return (
    <Loaded state={info} label="Opening the tutorial…">
      {(data) => data.practice ? <Coached practice={data.practice} preview={preview} next={next} />
        : <><PageHead title="Tutorial" subtitle="Nothing to practise on yet." />
          <Empty><p style={{ marginTop: 0 }}>The tutorial is not set up on this server yet: an admin makes it once with <code>stainid tutorial build</code>
            (a public dataset’s field and core, the same for every project).</p>
            <Link className="btn" to={next}>Back</Link></Empty></>}
    </Loaded>
  )
}

function Coached({ practice, preview, next }: { practice: NonNullable<TutorialInfo['practice']>; preview: boolean; next: string }) {
  const navigate = useNavigate()
  const field = useFetch<SessionDetail>(`/api/annotate/${practice.round2}`)
  const keys = field.data?.meta.keys
  const counted = keys?.groups?.find((g) => g.group === 'counted')
  const typed = (counted?.items.length ?? 0) > 1
  const hint = practice.hint?.text ?? ''
  const steps = useMemo(() => stepsFor(typed, counted ? (counted.items.length > 1 ? `a number (${counted.keys})` : counted.keys) : '1', hint), [typed, counted, hint])
  const count = useRef(0)  // events of the step so far (for steps done after several)
  const [index, setIndex] = useState(0)
  const [cheer, setCheer] = useState(false)
  useEffect(() => { count.current = 0 }, [index])
  const [error, setError] = useState<string | null>(null)
  const step = steps[Math.min(index, steps.length - 1)]
  const finished = index >= steps.length

  const remember = useCallback(async () => {
    if (preview) return  // the admin looking: nothing is remembered
    try { await rememberTutorial('objects') } catch (e) { setError((e as Error).message) }
  }, [preview])
  useEffect(() => { if (finished) void remember() }, [finished, remember])
  const at = useRef(0)
  at.current = index
  const onEvent = useCallback((event: RoundEvent) => {
    const i = at.current
    if (i >= steps.length || !steps[i].done.includes(event)) return
    count.current += 1
    if (count.current < (steps[i].times ?? 1)) return
    count.current = 0
    setIndex(i + 1); setCheer(true)
  }, [steps])
  useEffect(() => { if (!cheer) return; const timer = setTimeout(() => setCheer(false), 1200); return () => clearTimeout(timer) }, [cheer])
  const leave = () => { if (preview) window.history.length > 1 ? window.history.back() : window.close(); else navigate(next) }
  const skip = async () => { await remember(); leave() }

  const coach = (
    <aside className="card tutorial-coach in-panel" role="dialog" aria-label="Tutorial">
      <div className="row small"><b>Tutorial</b><span className="muted">{preview ? 'preview — what annotators see; nothing is saved' : 'practice — nothing here is saved'}</span>
        <span className="spacer" />
        {!finished && <button className="link-btn small" onClick={skip}>{preview ? 'Close' : 'Skip tutorial'}</button>}</div>
      {finished ? <>
        <h3>You are ready</h3>
        <p className="small secondary">That is all: point, press a key, correct detection’s work, and save each field. You can see this again with “Tutorial” at the top.</p>
        {preview ? <button className="btn primary" onClick={leave}>Close the preview</button> : <Link className="btn primary" to={next}>Start labelling →</Link>}
      </> : <>
        <div className="tutorial-step small muted">Step {index + 1} of {steps.length}{cheer && <span className="good-text"> · well done ✓</span>}</div>
        <h3>{step.title}</h3>
        <p className="small">{step.text}</p>
        <div className="tutorial-dots" aria-hidden="true">{steps.map((s, i) => <span key={s.title} className={i < index ? 'done' : i === index ? 'now' : ''} />)}</div>
        <div className="row">
          <button className="btn small" disabled={index === 0} onClick={() => setIndex(index - 1)}>Back</button>
          <button className="link-btn small" onClick={() => setIndex(index + 1)}>I did it / next</button>
        </div>
      </>}
      <ErrorNote error={error ?? field.error} />
      {practice.credit && <p className="small muted tutorial-credit">{practice.credit}</p>}
    </aside>
  )
  return (
    <div className="tutorial">
      {!field.data ? <Loading label="Opening the practice field…" />
        : <FixField key="practice-field" detail={field.data} reload={() => undefined} tile={field.data.meta.fields[0] ?? practice.tile_id} onEvent={onEvent} coach={coach}
          onFinished={() => setIndex(steps.length)} />}
    </div>
  )
}

/* ---------- the tutorial of "Mark bad areas" ---------- */

interface AreaStep { title: string; text: string; done: AreaEvent[] }
/** The steps; with one stain in the practice core (`single`) there is no stain to switch to, and Enter both ticks the
 *  stain and saves the core. */
function areaSteps(single: boolean): AreaStep[] {
  if (!single) return AREA_STEPS
  const steps = AREA_STEPS.filter((s) => !s.done.includes('stain') && !s.done.includes('saved'))
  return steps.map((s) => s.done.includes('type') ? { ...s, text: `You see one core at a time with each of its stains as a tab above the picture (one stain in this example). ${s.text}` }
    : s.done.includes('fine') ? { ...s, title: 'Save the core', text: 'When the core looks right — painted, or fine as it is — press Enter (or “Save and continue”): it is saved and the next core opens. With several stains, each stain gets its tick first. (In this practice nothing is kept.)', done: ['fine', 'saved'] }
      : s)
}
const AREA_STEPS: AreaStep[] = [
  { title: 'One core, all its stains', text: 'You see one core at a time with each of its stains (the tabs above the picture). Hatched parts are each type’s area as it stands: what detection leaves out, with everyone’s edits (detection’s own areas, what the analysis uses, do not change here). Press ] (or click a tab) to look at the next stain, [ to go back.', done: ['stain'] },
  { title: 'Choose a type', text: 'Press 1–4 to choose what is wrong: 1 fold or tissue damage, 2 artifact (scratch, bubble, debris), 3 no tissue, 4 out of focus or bad staining. The brush takes its colour, and that type’s area shows strongly as one region: detection’s with everyone’s edits.', done: ['type'] },
  { title: 'Paint a bad area', text: 'Drag over a part that should not be counted: it is added to the type’s area (a stroke touching an area of that type extends it). The wheel zooms; the middle mouse button moves the picture.', done: ['paint'] },
  { title: 'Brush size', text: 'Press Tab for a smaller brush, Shift+Tab for a bigger one — the same keys as in Fix the field. The ring under the mouse shows its size.', done: ['size'] },
  { title: 'Take a part out', text: 'Hold Shift and drag over a part of the chosen type’s area that is fine after all, also an area detection found: it is taken out of that type.', done: ['clear'] },
  { title: 'Undo', text: 'Press Z to take back your last stroke.', done: ['undo'] },
  { title: 'A clean look', text: 'Press Space to hide everything drawn over the tissue, and Space again to bring it back.', done: ['overlays'] },
  { title: 'This stain is done', text: 'When a stain looks right — painted, or fine as it is — press Enter. It gets a tick and the next stain opens.', done: ['fine'] },
  { title: 'Save the core', text: 'Press Enter on the last stain (or “Save and continue”) once every stain has a tick: the core is saved and the next one opens. (In this practice nothing is kept.)', done: ['saved'] },
]

function AreasCoached({ preview, next }: { preview: boolean; next: string }) {
  const navigate = useNavigate()
  const detail = useFetch<SessionDetail>(`/api/annotate/${PRACTICE_AREAS}`)
  const meta = detail.data?.meta as { stains_of?: Record<string, string[]>; credit?: string } | undefined
  const single = Object.values(meta?.stains_of ?? {})[0]?.length === 1
  const steps = useMemo(() => areaSteps(single), [single])
  const [index, setIndex] = useState(0)
  const [cheer, setCheer] = useState(false)
  const step = steps[Math.min(index, steps.length - 1)]
  const finished = index >= steps.length
  const remember = useCallback(() => {
    if (preview) return  // the admin looking: nothing is remembered
    void rememberTutorial('areas').catch(() => undefined)
  }, [preview])
  useEffect(() => { if (finished) remember() }, [finished, remember])
  const at = useRef(0)
  at.current = index
  const onEvent = useCallback((event: AreaEvent) => {
    const i = at.current
    if (i < steps.length && steps[i].done.includes(event)) { at.current = i + 1; setIndex(i + 1); setCheer(true) }
  }, [steps])
  useEffect(() => { if (!cheer) return; const timer = setTimeout(() => setCheer(false), 1200); return () => clearTimeout(timer) }, [cheer])
  const leave = () => { if (preview) window.history.length > 1 ? window.history.back() : window.close(); else navigate(next) }
  const coach = (
    <aside className="card tutorial-coach in-panel" role="dialog" aria-label="Tutorial">
      <div className="row small"><b>Tutorial</b><span className="muted">{preview ? 'preview — what annotators see; nothing is saved' : 'practice — nothing here is saved'}</span>
        <span className="spacer" />
        {!finished && <button className="link-btn small" onClick={() => { remember(); leave() }}>{preview ? 'Close' : 'Skip tutorial'}</button>}</div>
      {finished ? <>
        <h3>You are ready</h3>
        <p className="small secondary">Look at each stain, paint what should not be counted, press Enter for each stain, and the core is saved. The areas you mark
          teach and test the bad-area detection; the analysis uses detection’s own areas.</p>
        {preview ? <button className="btn primary" onClick={leave}>Close the preview</button> : <Link className="btn primary" to={next}>Start →</Link>}
      </> : <>
        <div className="tutorial-step small muted">Step {index + 1} of {steps.length}{cheer && <span className="good-text"> · well done ✓</span>}</div>
        <h3>{step.title}</h3>
        <p className="small">{step.text}</p>
        {index === 0 && <p className="small secondary">The areas you mark teach and test the bad-area detection; the analysis uses detection’s own areas.</p>}
        <div className="tutorial-dots" aria-hidden="true">{steps.map((s, i) => <span key={s.title} className={i < index ? 'done' : i === index ? 'now' : ''} />)}</div>
        <div className="row">
          <button className="btn small" disabled={index === 0} onClick={() => setIndex(index - 1)}>Back</button>
          <button className="link-btn small" onClick={() => setIndex(index + 1)}>I did it / next</button>
        </div>
      </>}
      {meta?.credit && <p className="small muted tutorial-credit">{meta.credit}</p>}
    </aside>
  )
  return (
    <div className="tutorial">
      {detail.error ? <><PageHead title="Tutorial" subtitle="Nothing to practise on yet." />
        <Empty><p style={{ marginTop: 0 }}>The tutorial is not set up on this server yet: an admin makes it once with <code>stainid tutorial build</code>.</p><ErrorNote error={detail.error} />
          <Link className="btn" to={next}>Back</Link></Empty></>
        : !detail.data ? <Loading label="Opening the practice core…" />
          : <MarkAreas detail={detail.data} onEvent={onEvent} coach={coach} onFinished={() => setIndex((i) => Math.max(i, steps.length))} />}
    </div>
  )
}
