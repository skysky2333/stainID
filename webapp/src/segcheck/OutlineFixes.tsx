/* "Outline fixes" on the Models page: per stain, the outlines people changed in "Fix the field" (a precise edit, a split
   or a merge) against each outline detection had for those objects — the stain outline and Segment Anything's small,
   medium and large masks, and the one shown first — as IoU and area ratio, with how often each was the closest
   (labels/outline_fixes.py). The evidence for each stain's default outline (Run page); never changes it. An indication,
   not an accuracy measure: only objects people chose to fix count. */
import { Link } from 'react-router-dom'
import { apiUrl } from '../api'
import { Empty, Loaded, Term } from '../components'
import { useFetch } from '../hooks'

interface OutlineOption { id: string; name: string; compared: number; iou_median: number | null; iou_mean: number | null; area_ratio_median: number | null; closest: number | null }
interface StainFixes {
  stain: string; objects: number; edited: number; switched?: number; split: number; merge: number; people: string[]; options: OutlineOption[]
  picked: Record<string, number>; default: 'stain' | 'sam'; hint: { option: string; text: string } | null
}

const OPTION_NAME: Record<string, string> = { stain: 'stain outline', sam: 'Segment Anything', sam_small: 'SAM small', sam_medium: 'SAM medium', sam_large: 'SAM large' }
const num = (v: number | null, digits = 2) => (v == null ? '–' : v.toFixed(digits))

export function OutlineFixes() {
  const state = useFetch<{ stains: StainFixes[] }>('/api/models/outline-fixes', 60000)
  return (
    <section className="card">
      <div className="row"><h2 style={{ margin: 0 }}>Outline fixes</h2><span className="spacer" />
        {state.data?.stains.some((s) => s.objects > 0) && <a className="btn small" href={apiUrl('/api/models/outline-fixes.csv')}>Download the fixed outlines (CSV)</a>}</div>
      <p className="secondary small" style={{ marginTop: 4 }}>How close each of detection’s outlines comes to the outlines people fixed in <b>Fix the field</b> (a precise
        edit, a split or a merge). It shows which outline to draw first for each stain (the <Link to="/workflow">Run page</Link> setting) and keeps the fixed outlines to
        train a better outline later. An indication, not an accuracy measure: only the objects people chose to fix count.</p>
      <Loaded state={state} label="Comparing the fixed outlines…">
        {(data) => data.stains.length === 0 ? <Empty>No stain has detection’s candidates yet.</Empty> : (
          <div className="segcheck-stains">{data.stains.map((s) => <StainFixesCard key={s.stain} s={s} />)}</div>
        )}
      </Loaded>
    </section>
  )
}

function StainFixesCard({ s }: { s: StainFixes }) {
  const kinds = [[s.edited, 'edited precisely'], [s.switched ?? 0, 'switched to the other outline (Tab)'], [s.split, 'split'], [s.merge, 'merged']].filter(([n]) => n).map(([n, w]) => `${(n as number).toLocaleString()} ${w}`)
  const picked = Object.entries(s.picked).filter(([, n]) => n)
  const best = Math.max(...s.options.filter((o) => o.id !== 'shown').map((o) => o.iou_median ?? -1))
  return (
    <div className="segcheck-stain">
      <div className="row"><h3 style={{ margin: 0 }}>{s.stain}</h3><span className="spacer" />
        <span className="small muted">drawn first: {s.default === 'sam' ? 'Segment Anything' : 'the stain outline'}</span></div>
      {s.objects === 0 ? <p className="small muted">No outline changed in “Fix the field” yet.</p> : <>
        <div className="small secondary">{s.objects.toLocaleString()} object{s.objects === 1 ? '' : 's'} with an outline people changed ({kinds.join(', ')})
          {s.people.length > 0 && ` by ${s.people.length} ${s.people.length === 1 ? 'person' : 'people'}`}</div>
        <table className="table compact" style={{ marginTop: 6 }}>
          <thead><tr><th>Detection’s outline</th><th title="intersection over union with the person's outline, median over the objects"><Term term="IoU">IoU</Term> (median)</th>
            <th title="detection's area over the person's, median: below 1 too small, above 1 too large">area ratio</th><th>closest for</th></tr></thead>
          <tbody>{s.options.map((o) => (
            <tr key={o.id} className={o.id === 'shown' ? 'muted' : o.iou_median === best && best >= 0 ? 'is-best' : undefined}>
              <td>{o.id === 'shown' ? <i>shown first</i> : o.name}</td><td>{num(o.iou_median)}</td><td>{num(o.area_ratio_median)}</td>
              <td>{o.closest == null ? '' : `${o.closest.toLocaleString()} of ${s.objects.toLocaleString()}`}</td>
            </tr>
          ))}</tbody>
        </table>
      </>}
      {picked.length > 0 && <p className="small secondary" style={{ margin: '4px 0 0' }}>Switched to the other outline (Tab):{' '}
        {picked.map(([k, n]) => `${OPTION_NAME[k] ?? k} ${n.toLocaleString()}×`).join(', ')}.</p>}
      {s.hint && <p className="small" style={{ margin: '6px 0 0' }}><b>Hint:</b> {s.hint.text}. Change it on the <Link to="/workflow">Run page</Link> if it holds up.</p>}
    </div>
  )
}
