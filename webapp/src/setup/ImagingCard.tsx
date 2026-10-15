import { useState } from 'react'
import type { ImagesOverview, StainFamily, StainSetting } from '../api'
import { api, apiUrl, enc } from '../api'
import { ErrorNote, withError } from '../components'
import { Advanced, plural, SavedNote } from './parts'

type Draft = Record<string, { family: StainFamily; chromogen: string; others: string[] }>

const draftOf = (stains: StainSetting[]): Draft =>
  Object.fromEntries(stains.map((s) => [s.stain, { family: s.family || '', chromogen: s.chromogen || 'DAB', others: s.others ?? [] }]))

/** How the images were stained and scanned: what each stain is (which of stainID's models reads it), the colour it was
developed in (and a second colour on the same slide), and the pixel size. Rescaling is under Advanced. */
export default function ImagingCard({ data, onSaved }: { data: ImagesOverview; onSaved: () => void }) {
  const [draft, setDraft] = useState<Draft>(() => draftOf(data.stains))
  const [pixel, setPixel] = useState(data.imaging.native_pixel_size_um?.toString() ?? '')
  const [rescale, setRescale] = useState(data.imaging.rescale)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [saved, setSaved] = useState(false)
  const stains = data.stains.filter((s) => s.images > 0 || s.set)
  const sample = (stain: string) => data.rows.find((r) => r.stain === stain)?.image_id
  const set = (stain: string, change: Partial<Draft[string]>) => { setDraft({ ...draft, [stain]: { ...draft[stain], ...change } }); setSaved(false) }
  const dirty = JSON.stringify(draft) !== JSON.stringify(draftOf(data.stains)) || pixel !== (data.imaging.native_pixel_size_um?.toString() ?? '')
    || rescale !== data.imaging.rescale
  const save = () => withError(async () => {
    setBusy(true)
    try {
      const number = pixel.trim() ? Number(pixel.replace(',', '.')) : null
      if (number !== null && !(number > 0)) throw new Error('The pixel size must be a number of micrometres, such as 0.5')
      await api.put('/api/setup/images/imaging', { native_pixel_size_um: number, rescale, stains: draft })
      setSaved(true); onSaved()
    } finally { setBusy(false) }
  }, setError)
  const colour = (id: string) => data.chromogens.find((c) => c.id === id)?.label ?? id

  return (
    <div className="card imaging-card">
      <h3>How the {data.kind === 'slides' ? 'slides' : 'images'} were stained</h3>
      <p className="secondary small">stainID's models know three kinds of stain. Say which kind each of yours is and the colour it was developed in;
        an image in another colour than brown is turned into brown-on-blue before it is analysed, so the models read it as they read DAB.</p>
      <div className="imaging-stains">
        {stains.map((s) => {
          const own = draft[s.stain] ?? { family: '', chromogen: 'DAB', others: [] }
          const image = sample(s.stain)
          return (
            <div key={s.stain} className="imaging-stain">
              <div className="imaging-stain-name"><b>{s.stain || 'No stain given'}</b><span className="muted small">{plural(s.images, data.kind === "slides" ? "slide" : "image")}</span></div>
              <label className="inline-field small">Kind of stain
                <select value={own.family} onChange={(e) => set(s.stain, { family: e.target.value as StainFamily })}>
                  <option value="">choose…</option>
                  {Object.entries(data.families).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
                </select></label>
              {own.family !== 'ignore' && <>
                <label className="inline-field small">Colour
                  <select value={own.chromogen} onChange={(e) => set(s.stain, { chromogen: e.target.value, others: own.others.filter((o) => o !== e.target.value) })}>
                    {data.chromogens.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
                  </select></label>
                <label className="inline-field small">Second colour on these slides
                  <select value={own.others[0] ?? ''} onChange={(e) => set(s.stain, { others: e.target.value ? [e.target.value] : [] })}>
                    <option value="">none</option>
                    {data.chromogens.filter((c) => c.id !== own.chromogen).map((c) => <option key={c.id} value={c.id}>{c.label} (left out)</option>)}
                  </select></label>
              </>}
              {own.others.length > 0 && <p className="small secondary imaging-note">Two stains on one slide: only the {colour(own.chromogen)} one ({s.stain}) is analysed;
                the {own.others.map(colour).join(' and ')} one is left out.</p>}
              {image && own.family && own.family !== 'ignore' && (
                <figure className="imaging-preview">
                  <img src={apiUrl(`/api/setup/images/${enc(image)}/picture.jpg`)} alt={`${image} as scanned`} loading="lazy" />
                  <img src={apiUrl(`/api/setup/images/${enc(image)}/picture.jpg?analysis=1&v=${enc(JSON.stringify(data.stains.find((x) => x.stain === s.stain)))}`)}
                    alt={`${image} as stainID reads it`} loading="lazy" />
                  <figcaption className="small muted">As scanned, and as stainID reads it (after saving): the stain it analyses in brown.</figcaption>
                </figure>
              )}
            </div>
          )
        })}
      </div>
      <label className="inline-field small">Pixel size of the {data.kind === 'slides' ? 'scans' : 'images'} (µm)
        <input type="text" inputMode="decimal" value={pixel} placeholder="read from each file" onChange={(e) => { setPixel(e.target.value); setSaved(false) }} /></label>
      <p className="small muted">Leave it empty when the files or the image list say it. The models work at {data.imaging.analysis_pixel_size_um} µm per pixel; other
        images are enlarged or reduced to that.</p>
      <Advanced hint="when images are rescaled">
        <div className="radio-list small">
          <label><input type="radio" checked={rescale === 'on_import'} onChange={() => { setRescale('on_import'); setSaved(false) }} /> Convert each image once, when the
            images are prepared, and keep the result among the app's pictures (faster; takes disk space, given back when the space for pictures runs out)</label>
          <label><input type="radio" checked={rescale === 'on_the_fly'} onChange={() => { setRescale('on_the_fly'); setSaved(false) }} /> Convert each time an image is read
            (no extra disk space; slower; the choice for whole slides)</label>
        </div>
        <p className="small muted">Own stain colours (measured from an area of pure stain) can be given as <code>vectors</code> under <code>imaging.stains</code> in
          stainid.yaml. After changing anything here, run <b>Prepare images</b> again (Run → Run everything does it).</p>
      </Advanced>
      <div className="row">
        <span className="spacer" />
        <SavedNote shown={saved && !dirty} />
        <button className="btn" disabled={busy || !dirty} onClick={save}>{busy ? 'Saving…' : 'Save'}</button>
      </div>
      <ErrorNote error={error} />
    </div>
  )
}
