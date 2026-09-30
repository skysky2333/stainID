import { Link } from 'react-router-dom'
import { Empty, ErrorNote, PageHead } from '../components'
import { useFetch } from '../hooks'

/** The fitted core grid drawn on every slide: the quick visual check after "Find cores". */
export default function Grids() {
  const grids = useFetch<string[]>('/api/grids')
  return (
    <>
      <PageHead title="Core grid check" subtitle={<>Every circle should sit on a core. If a grid is shifted, check the rows / columns in <Link to="/settings">Settings</Link> and run Find cores again with “Redo all slides”.</>} />
      <ErrorNote error={grids.error} />
      {grids.data && !grids.data.length && <Empty>No grid pictures yet — run <Link to="/workflow#dearray">Find cores</Link>.</Empty>}
      <div className="grid grid-2">
        {(grids.data ?? []).map((name) => (
          <a key={name} className="card grid-shot" href={`/api/grids/${encodeURIComponent(name)}`} target="_blank" rel="noreferrer">
            <img src={`/api/grids/${encodeURIComponent(name)}`} alt={name} loading="lazy" />
            <div className="small secondary">{name.replace(/\.jpg$/, '').replaceAll('_', ' ')}</div>
          </a>
        ))}
      </div>
    </>
  )
}
