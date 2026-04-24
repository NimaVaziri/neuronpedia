import { CircuitCausalityResult } from './causal-validation';

export default function CircuitCausalityResultView({
  validation,
  compact = false,
}: {
  validation: CircuitCausalityResult;
  compact?: boolean;
}) {
  return (
    <div className="flex flex-col gap-y-1">
      <div className={`flex ${compact ? 'flex-col gap-y-1' : 'items-center gap-x-3'} text-[11px]`}>
        <span className="group relative">
          <span className={validation.necessity > 0.05 ? 'font-medium text-red-600' : 'text-slate-400'}>
            Necessity (higher drop = more necessary): {(validation.baselineProb * 100).toFixed(0)}% -&gt;{' '}
            {((validation.baselineProb - validation.necessity) * 100).toFixed(0)}%
            <span className="ml-1 text-[10px]">
              ({validation.necessity > 0 ? '-' : ''}
              {(validation.necessity * 100).toFixed(1)}pp)
            </span>
          </span>
          <span className="pointer-events-none absolute bottom-full left-0 z-50 mb-1 hidden w-52 rounded bg-slate-800 px-2 py-1 text-[11px] text-white shadow-lg group-hover:block">
            How much the prediction drops when this circuit is removed. A large drop = causally important.
          </span>
        </span>
        <span className="group relative">
          <span className={validation.sufficiency > 0.5 ? 'font-medium text-emerald-600' : 'text-slate-400'}>
            Sufficiency (higher = more sufficient): {(validation.baselineProb * 100).toFixed(0)}% -&gt;{' '}
            {(validation.sufficiency * validation.baselineProb * 100).toFixed(0)}%
            <span className="ml-1 text-[10px]">({(validation.sufficiency * 100).toFixed(0)}% retained)</span>
          </span>
          <span className="pointer-events-none absolute bottom-full left-0 z-50 mb-1 hidden w-52 rounded bg-slate-800 px-2 py-1 text-[11px] text-white shadow-lg group-hover:block">
            How much of the prediction survives when only this circuit is kept. High = circuit alone is sufficient.
          </span>
        </span>
      </div>
      <div className={`grid ${compact ? 'grid-cols-1 gap-y-1.5' : 'grid-cols-3 gap-x-2'} text-[10px]`}>
        <div className="min-w-0">
          <div className="font-semibold text-slate-500">Baseline</div>
          {validation.baselineLogits.map((l, j) => (
            <div key={j} className="flex justify-between gap-x-1 text-slate-500">
              <span className="truncate">{l.token}</span>
              <span className="font-mono">{(l.prob * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
        <div className="min-w-0">
          <div className="font-semibold text-red-500">Ablated (necessity)</div>
          {validation.necessityLogits.map((l, j) => (
            <div key={j} className="flex justify-between gap-x-1 text-red-600">
              <span className="truncate">{l.token}</span>
              <span className="font-mono">{(l.prob * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
        <div className="min-w-0">
          <div className="font-semibold text-amber-600">Complement (sufficiency)</div>
          {validation.sufficiencyLogits.map((l, j) => (
            <div key={j} className="flex justify-between gap-x-1 text-amber-700">
              <span className="truncate">{l.token}</span>
              <span className="font-mono">{(l.prob * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
