from pathlib import Path
import shutil
import warnings
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA




BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / 'data'
OUT = BASE / 'results_6'
PAGE_START = 0
N_PAGES = 10
N_DAYS = 400
TEST_DAYS = 30
HORIZONS = ( 15, 30 )
FOLDS = 3

ORDERS = [
    (p, d, q)
    for p in range(9)
    for d in (0, 1, 2)
    for q in range(9)
]


TRANSFORMS = ( 'raw', 'log1p' )
WINDOWS = ( None, 240 )

plt.rcParams['font.family'] = 'DejaVu Sans'





def get_dataset ():
    archive = DATA_DIR / 'train_1.csv.zip'
    csv = DATA_DIR / 'train_1.csv'
    if archive.exists ():
        return archive
    if csv.exists ():
        return csv
    try:
        import kagglehub
        print ( 'Downloading dataset from Kaggle...', flush=True )
        downloaded = Path ( kagglehub.competition_download (
            'web-traffic-time-series-forecasting', 'train_1.csv.zip'))
        if downloaded.is_dir ():
            downloaded = downloaded / 'train_1.csv.zip'
        if not downloaded.is_file ():
            raise FileNotFoundError ( f'Archive was not downloaded: {downloaded}' )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if downloaded.resolve() != archive.resolve():
            shutil.copy2(downloaded, archive)
        return archive
    except Exception as exc:
        raise RuntimeError( 'Kaggle download failed.' ) from exc


def load_pages ( ):
    source = get_dataset ( )
    kwargs = { 'skiprows': range(1, PAGE_START + 1), 'nrows': N_PAGES }
    if source.suffix == '.zip':
        with zipfile.ZipFile ( source ) as archive:
            with archive.open ( 'train_1.csv' ) as handle:
                frame = pd.read_csv ( handle, **kwargs )
    else:
        frame = pd.read_csv ( source, **kwargs )
    if len ( frame ) != N_PAGES:
        raise ValueError ( f'Expected {N_PAGES} pages, found {len(frame)}' )
    dates = pd.to_datetime ( frame.columns[1:] )
    pages = []

    for i, row in frame.iterrows ( ) :
        values = pd.to_numeric ( row.iloc[1:], errors='coerce' ).to_numpy ( dtype=float )
        series = pd.Series ( values, index=dates, name='Visitors' ).iloc[-N_DAYS:]
        series = series.asfreq( 'D' ).ffill().fillna ( 0.0 )

        if len ( series ) < ( FOLDS + 1 ) * TEST_DAYS + 70:
            raise ValueError ( f'Insufficient data for page {PAGE_START + i + 1}' )
        
        if not np.isfinite ( series.to_numpy() ).all() or (series < 0).any():
            raise ValueError ( f'Invalid data for page {PAGE_START + i + 1} ')
        
        pages.append ( ( PAGE_START + i + 1, str ( row.iloc[0] ), series ) )

    return pages


def fit_model ( history, order, transform, window ) :
    if window is not None:
        history = history.iloc[-window:]
    values = np.log1p(history) if transform == 'log1p' else history.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ARIMA(values, order=order).fit(method_kwargs={'maxiter': 100})


def restore_scale ( values, transform ):
    prediction = np.asarray(values, dtype=float)
    if transform == 'log1p':
        prediction = np.expm1(np.clip(prediction, -20, 20))
    return np.maximum(prediction, 0.0)


def forecast_fixed(model, horizon, transform):
    return restore_scale(model.forecast(steps=horizon), transform)


def forecast_one_step(model, actual, transform):
    predicted = []
    for observed in np.asarray(actual, dtype=float):
        predicted.append(forecast_fixed(model, 1, transform)[0])
        update = float(np.log1p(observed) if transform == 'log1p' else observed)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = model.append([update], refit=False)
    return np.asarray(predicted)


def metrics(actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    error = np.abs(a - p)
    return (float(np.mean(error)),
            float(np.sqrt(np.mean(error ** 2))),
            float(np.mean(200 * error / np.maximum(np.abs(a) + np.abs(p), 1e-8))))


def validate(train, order, transform, window):
    scores = []
    for fold in range(FOLDS):
        start = len(train) - (FOLDS - fold) * TEST_DAYS
        history = train.iloc[:start]
        target = train.iloc[start:start + TEST_DAYS]
        model = fit_model(history, order, transform, window)
        predicted = forecast_fixed(model, len(target), transform)
        score = metrics(target, predicted)
        if not np.isfinite(score).all():
            raise ValueError('Invalid metrics')
        scores.append(score)
    return np.mean(scores, axis=0)


def select_model ( train ) :
    results = []
    for order in ORDERS:
        for transform in TRANSFORMS:
            for window in WINDOWS:
                try:
                    mae, rmse, smape = validate ( train, order, transform, window )
                    results.append ( ( order, transform, window, mae, rmse, smape ) )
                except ( ValueError, np.linalg.LinAlgError, RuntimeError ):
                    continue
    if not results:
        raise RuntimeError ( 'No ARIMA configuration completed validation' )
    return min ( results, key=lambda row: row[4] )


def train_page ( page_id, page_name, series ):
    train, test = series.iloc [ :-TEST_DAYS ], series.iloc [ -TEST_DAYS: ]
    order, transform, window, _, validation_rmse, _ = select_model ( train )
    model = fit_model ( train, order, transform, window )
    fixed = forecast_fixed ( model, len ( test ), transform )
    step = forecast_one_step ( model, test, transform )
    results = []
    for horizon in HORIZONS:
        target = test.iloc[:horizon]
        fixed_part = fixed[:horizon]
        step_part = step[:horizon]
        fixed_mae, fixed_rmse, fixed_smape = metrics ( target, fixed_part )
        step_mae, step_rmse, step_smape = metrics ( target, step_part )
        results.append ( {
            'page': page_id, 'name': page_name, 'order': order,
            'transform': transform, 'window': window, 'cv_rmse': validation_rmse,
            'horizon': horizon,
            'fixed_mae': fixed_mae, 'fixed_rmse': fixed_rmse, 'fixed_smape': fixed_smape,
            'step_mae': step_mae, 'step_rmse': step_rmse, 'step_smape': step_smape,
            'series': series, 'test': target, 'fixed': fixed_part, 'step': step_part,
        } )
    return results


def save_results ( results ):
    fields = ['page', 'name', 'order', 'transform', 'window', 'cv_rmse', 'horizon',
              'fixed_mae', 'fixed_rmse', 'fixed_smape',
              'step_mae', 'step_rmse', 'step_smape']
    table = pd.DataFrame ( [{key: r[key] for key in fields} for r in results] )
    table.to_csv ( OUT / 'results.csv', index=False )


def plot_forecasts ( results, horizon ) :
    fig, axes = plt.subplots ( 5, 2, figsize=(16, 16), sharex=False )
    for ax, r in zip ( axes.flat, results ):
        series, test = r['series'], r['test']
        history = series.iloc[-(TEST_DAYS + 60):-TEST_DAYS]
        actual = pd.concat ( [history, test] )
        ax.plot ( actual.index, actual, lw=1, label='Actual' )
        ax.plot ( test.index, r['fixed'], lw=1.8, label=f'{horizon}-day' )
        ax.plot ( test.index, r['step'], lw=1.2, label='One-step' )
        ax.axvline ( test.index[0], ls='--', lw=0.8, color='gray' )
        ax.set_title ( f"Page {r['page']} | ARIMA{r['order']}", fontsize=10 )
        ax.set_ylabel ( 'Visitors' )
        ax.tick_params ( axis='x', labelrotation=25, labelsize=8 )
        ax.grid ( alpha=0.2 )
    axes.flat[0].legend ( fontsize=8 )
    fig.suptitle ( f'Web traffic forecasts: {horizon}-day horizon', fontsize=14 )
    fig.tight_layout( rect=(0, 0, 1, 0.98) )
    fig.savefig ( OUT / f'forecasts_{horizon}.png', dpi=160 )
    plt.close ( fig )


def plot_comparison ( results, horizon ):
    x = np.arange ( len ( results ) )
    width = 0.38
    fig, ax = plt.subplots ( figsize=( 11, 5 ) )
    ax.bar ( x - width / 2, [r['fixed_rmse'] for r in results], width, label=f'{horizon}-day' )
    ax.bar ( x + width / 2, [r['step_rmse'] for r in results], width, label='One-step' )
    ax.set_xticks ( x, [f"Page {r['page']}" for r in results] )
    ax.set_ylabel ( 'Test RMSE (visitors)' )
    ax.set_title ( f'Forecast quality by page: {horizon}-day horizon' )
    ax.legend ( )
    fig.tight_layout ( )
    fig.savefig ( OUT / f'comparison_{horizon}.png', dpi=160 )
    plt.close ( fig )


def plot_parameters ( results ) :
    fig, ax = plt.subplots ( figsize=( 11, 5 ) )
    x = np.arange ( len ( results ) )
    ax.bar ( x, [r['cv_rmse'] for r in results ] )
    ax.set_xticks ( x, [f"Page {r['page']}\n{r['order']}" for r in results] )
    ax.set_ylabel ( 'Mean 30-day validation RMSE' )
    ax.set_title ( 'Selected ARIMA configurations for each page' )
    fig.tight_layout ( )
    fig.savefig ( OUT / 'parameters.png', dpi=160 )
    plt.close ( fig )


def main():
    pages = load_pages ( )
    OUT.mkdir ( parents=True, exist_ok=True )
    results = []
    print( f'Pages: {len(pages)} | Days per page: {N_DAYS} | Horizons: {HORIZONS}', flush=True )

    for page_id, page_name, series in pages:
        page_results = train_page ( page_id, page_name, series )
        results.extend ( page_results )
        for result in page_results:
            print(f"Page {page_id:2d}: ARIMA{result['order']} | "
                  f"{result['horizon']}-day RMSE={result['fixed_rmse']:.2f} | "
                  f"One-step RMSE={result['step_rmse']:.2f}", flush=True)

    save_results ( results )
    for horizon in HORIZONS:
        horizon_results = [r for r in results if r['horizon'] == horizon]
        plot_forecasts ( horizon_results, horizon )
        plot_comparison ( horizon_results, horizon )
    plot_parameters ( [r for r in results if r['horizon'] == TEST_DAYS] )
    print ( f'Results saved to {OUT}/', flush=True )


if __name__ == '__main__':
    main ( )
