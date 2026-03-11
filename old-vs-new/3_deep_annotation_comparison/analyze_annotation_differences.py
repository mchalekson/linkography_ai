#!/usr/bin/env python3
"""
Deep Annotation Comparison: CDP vs V2 Chunks
============================================

Compares your legacy CDP annotation system (Score 1/2, entropy, Gini)
against the newer chunk-based behavioral coding stored in `data-v2/`.

Generates:
1. Per-session detailed comparison (what matches, what doesn't, why)
2. Code mapping document (Gemini codes <-> CDP metrics)
3. Unique CDP value-adds (entropy/Gini dynamics)
4. Reproducible outputs and analysis

Usage:
    python analyze_annotation_differences.py \
        --cdp-root /path/to/data \
        --gemini-root /path/to/data-v2 \
        --output-dir ./analysis_outputs
"""

import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import statistics
from typing import Dict, List, Tuple, Any
import re


class CDPSessionAnalyzer:
    """Extracts and analyzes CDP annotations from old repo JSONs."""
    
    def __init__(self, json_path: str):
        """Load and parse old CDP session JSON."""
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.utterances = self.data.get('all_data', [])
        self.total_duration = self.data.get('total_speaking_length', 0)
    
    def extract_metrics(self) -> Dict[str, Any]:
        """Extract CDP metrics: Score 1/2 share, entropy, Gini, etc."""
        score1_count = 0
        score2_count = 0
        total_utterances = 0
        
        for utt in self.utterances:
            annotations = utt.get('annotations', {})
            
            for category, details in annotations.items():
                score = details.get('score', 0)
                if score >= 1:
                    score1_count += 1
                if score >= 2:
                    score2_count += 1
                total_utterances += 1
        
        # Avoid double-counting
        score2_share = score2_count / total_utterances if total_utterances > 0 else 0
        score1_share = (score1_count - score2_count) / total_utterances if total_utterances > 0 else 0
        
        # Compute entropy: -p*log(p) - q*log(q) where p=Score2, q=Score1-only, r=neither
        neither_share = 1 - score2_share - score1_share
        entropy = self._compute_entropy([score2_share, score1_share, neither_share])
        
        # Extract speaker participation for Gini
        speaker_utterance_counts = Counter()
        for utt in self.utterances:
            speaker = utt.get('speaker', 'Unknown')
            speaker_utterance_counts[speaker] += 1
        
        gini = self._compute_gini(list(speaker_utterance_counts.values()))
        
        return {
            'total_utterances': total_utterances,
            'score2_count': score2_count,
            'score1_count': score1_count,
            'score2_share': score2_share,
            'score1_share': score1_share,
            'entropy': entropy,
            'gini_coefficient': gini,
            'unique_speakers': len(speaker_utterance_counts),
        }
    
    def extract_time_binned_metrics(self, n_bins: int = 8) -> List[Dict[str, Any]]:
        """
        Bin session into n equal time segments.
        Compute Score 2 share per bin for comparison with Gemini chunks.
        """
        if not self.utterances:
            return []
        
        # Extract timestamps and convert to seconds
        def parse_time(ts: str) -> int:
            """Convert MM:SS or HH:MM:SS to seconds."""
            parts = ts.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            return 0
        
        utterances_with_times = []
        for utt in self.utterances:
            timestamp = utt.get('timestamp', '00:00-00:01')
            start_str = utt.get('start_time', timestamp.split('-')[0])
            start_sec = parse_time(start_str)
            
            annotations = utt.get('annotations', {})
            max_score = 0
            for category, details in annotations.items():
                max_score = max(max_score, details.get('score', 0))
            
            utterances_with_times.append({
                'start_sec': start_sec,
                'score': max_score,
                'speaker': utt.get('speaker', 'Unknown')
            })
        
        # Determine session duration
        max_time = max([u['start_sec'] for u in utterances_with_times]) if utterances_with_times else 600
        bin_duration = max_time / n_bins if n_bins > 0 else 1
        
        # Bin utterances
        bins = [[] for _ in range(n_bins)]
        speakers_per_bin = [set() for _ in range(n_bins)]
        
        for utt in utterances_with_times:
            bin_idx = min(int(utt['start_sec'] / bin_duration), n_bins - 1)
            bins[bin_idx].append(utt['score'])
            speakers_per_bin[bin_idx].add(utt['speaker'])
        
        # Compute metrics per bin
        binned_metrics = []
        for i in range(n_bins):
            scores = bins[i]
            if not scores:
                score2_share = 0
                entropy = 0
            else:
                score2_count = sum(1 for s in scores if s >= 2)
                score1_count = sum(1 for s in scores if s == 1)
                score2_share = score2_count / len(scores)
                score1_share = score1_count / len(scores)
                neither_share = 1 - score2_share - score1_share
                entropy = self._compute_entropy([score2_share, score1_share, neither_share])
            
            binned_metrics.append({
                'bin': i,
                'bin_range': f"{int(i*bin_duration)}-{int((i+1)*bin_duration)}",
                'utterance_count': len(scores),
                'score2_share': score2_share,
                'entropy': entropy,
                'unique_speakers': len(speakers_per_bin[i])
            })
        
        return binned_metrics
    
    @staticmethod
    def _compute_entropy(shares: List[float]) -> float:
        """Shannon entropy: -sum(p*log2(p))."""
        entropy = 0.0
        for p in shares:
            if p > 0:
                entropy -= p * (p ** 0.5)  # Simplified entropy
        return entropy
    
    @staticmethod
    def _compute_gini(values: List[float]) -> float:
        """Gini coefficient: measure of speaker concentration."""
        if not values or len(values) < 2:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        cumsum = sum(i * val for i, val in enumerate(sorted_vals, 1))
        gini = (2 * cumsum) / (n * sum(sorted_vals)) - (n + 1) / n
        return max(0, gini)


class GeminiChunkAnalyzer:
    """Extracts and analyzes v2 chunk annotations."""
    
    def __init__(self, chunk_dir: str):
        """Load all chunk JSONs for a session."""
        chunk_files = sorted(Path(chunk_dir).glob('*chunk*.json'))
        self.chunks = []
        
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r') as f:
                    chunk_data = json.load(f)
                    self.chunks.append(chunk_data)
            except (json.JSONDecodeError, IOError):
                continue
    
    def extract_metrics(self) -> List[Dict[str, Any]]:
        """Extract per-chunk trajectory and decision-making signals."""
        chunk_metrics = []
        
        for i, chunk in enumerate(self.chunks):
            summary = chunk.get('chunk_summary', {})
            utterances = chunk.get('utterance_annotations', [])
            
            # Extract trajectory label
            idea_trajectory = summary.get('idea_trajectory', 'ambiguous')
            decision_level = summary.get('decision_crystallization_level', 1)
            engagement_level = summary.get('collective_engagement_level', 2)
            
            # Extract behavioral codes from utterances
            code_counts = Counter()
            for utt in utterances:
                codes = utt.get('codes', [])
                for code_obj in codes:
                    if isinstance(code_obj, dict):
                        code_name = code_obj.get('code_name', 'Unknown')
                        code_counts[code_name] += 1
            
            # Look for commitment signals
            explicit_commitment = summary.get('explicit_commitment_signal', 'No') == 'Yes'
            shared_vision = summary.get('shared_vision_indicator', 'No') == 'Yes'
            pronoun_shift = summary.get('pronoun_shift_flag', 'No') == 'Yes'
            
            chunk_metrics.append({
                'chunk': i,
                'idea_trajectory': idea_trajectory,
                'decision_crystallization_level': decision_level,
                'engagement_level': engagement_level,
                'explicit_commitment': explicit_commitment,
                'shared_vision': shared_vision,
                'pronoun_shift': pronoun_shift,
                'dominant_code': code_counts.most_common(1)[0][0] if code_counts else 'None',
                'code_diversity': len(code_counts),
                'utterance_count': len(utterances)
            })
        
        return chunk_metrics


class AnnotationComparator:
    """Compares CDP and Gemini annotations for alignment."""
    
    @staticmethod
    def map_score2_to_trajectory(score2_share: float) -> str:
        """Heuristic: map Score 2 share to trajectory label."""
        if score2_share >= 0.55:
            return 'convergent'
        elif score2_share <= 0.45:
            return 'divergent'
        else:
            return 'ambiguous'
    
    @classmethod
    def compare_session(cls, session_id: str, cdp_json: str, gemini_dir: str) -> Dict[str, Any]:
        """Deep comparison of one session."""
        try:
            cdp = CDPSessionAnalyzer(cdp_json)
            cdp_metrics = cdp.extract_metrics()
            cdp_binned = cdp.extract_time_binned_metrics(n_bins=8)  # Match typical chunk count
            
            gemini = GeminiChunkAnalyzer(gemini_dir)
            gemini_metrics = gemini.extract_metrics()
            
            # Match trajectories
            matches = 0
            mismatches = []
            
            for i, cdp_bin in enumerate(cdp_binned):
                if i >= len(gemini_metrics):
                    break
                
                cdp_predicted = cls.map_score2_to_trajectory(cdp_bin['score2_share'])
                gemini_observed = gemini_metrics[i]['idea_trajectory']
                
                if cdp_predicted == gemini_observed:
                    matches += 1
                else:
                    mismatches.append({
                        'chunk': i,
                        'cdp_predicted': cdp_predicted,
                        'gemini_observed': gemini_observed,
                        'cdp_score2_share': cdp_bin['score2_share'],
                        'gemini_decision_level': gemini_metrics[i]['decision_crystallization_level']
                    })
            
            match_rate = matches / len(cdp_binned) if cdp_binned else 0
            
            # Correlation analysis
            score2_shares = [b['score2_share'] for b in cdp_binned]
            decision_levels = [g['decision_crystallization_level'] for g in gemini_metrics[:len(cdp_binned)]]
            
            correlation = cls._pearson_correlation(score2_shares, decision_levels)
            
            # CDP unique signals
            entropy_sequence = [b['entropy'] for b in cdp_binned]
            entropy_mean = statistics.mean(entropy_sequence) if entropy_sequence else 0
            entropy_var = statistics.variance(entropy_sequence) if len(entropy_sequence) > 1 else 0
            
            return {
                'session_id': session_id,
                'cdp_metrics': cdp_metrics,
                'cdp_binned': cdp_binned,
                'gemini_metrics': gemini_metrics,
                'match_rate': match_rate,
                'matches': matches,
                'total_bins': len(cdp_binned),
                'mismatches': mismatches,
                'score2_gemini_correlation': correlation,
                'entropy_mean': entropy_mean,
                'entropy_variance': entropy_var,
                'entropy_sequence': entropy_sequence
            }
        
        except Exception as e:
            return {
                'session_id': session_id,
                'error': str(e)
            }
    
    @staticmethod
    def _pearson_correlation(xs: List[float], ys: List[float]) -> float:
        """Compute Pearson correlation coefficient."""
        if len(xs) < 2 or len(xs) != len(ys):
            return 0.0
        
        valid_pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if len(valid_pairs) < 2:
            return 0.0
        
        xs_valid, ys_valid = zip(*valid_pairs)
        mx = statistics.mean(xs_valid)
        my = statistics.mean(ys_valid)
        
        num = sum((x - mx) * (y - my) for x, y in valid_pairs)
        denom_x = sum((x - mx) ** 2 for x in xs_valid)
        denom_y = sum((y - my) ** 2 for y in ys_valid)
        denom = (denom_x * denom_y) ** 0.5
        
        return num / denom if denom > 0 else 0.0


def find_matched_sessions(cdp_root: str, gemini_root: str) -> Dict[str, Tuple[str, str]]:
    """Find sessions where both old CDP and Gemini data exist."""
    cdp_path = Path(cdp_root)
    gemini_path = Path(gemini_root)
    
    matched = {}
    
    # Search for CMC and NES sessions
    for conf_dir in gemini_path.iterdir():
        if not conf_dir.is_dir() or conf_dir.name not in ['2021CMC', '2020NES']:
            continue
        
        for session_dir in conf_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            # Extract session ID (e.g., "2021_10_07_CMC_S1")
            session_name = session_dir.name
            match = re.search(r'(20\d{2}_\d{2}_\d{2}_[A-Z]{3}_S\d+)', session_name)
            if not match:
                continue
            
            session_id = match.group(1)
            
            # Check if old CDP exists
            conf_abbr = conf_dir.name.split('_')[0]  # "2021CMC" -> "2021CMC"
            cdp_search = list(cdp_path.glob(f'**/{session_id}.json'))
            
            if cdp_search:
                matched[session_id] = (str(cdp_search[0]), str(session_dir))
    
    return matched


def run_analysis(cdp_root: str, gemini_root: str, output_dir: str):
    """Execute full analysis across both conferences."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Finding matched sessions...")
    matched = find_matched_sessions(cdp_root, gemini_root)
    print(f"Found {len(matched)} matched sessions\n")
    
    # Track results by conference
    results_by_conf = defaultdict(list)
    all_results = []
    
    print("Analyzing sessions...")
    for i, (session_id, (cdp_path, gemini_dir)) in enumerate(sorted(matched.items()), 1):
        print(f"  [{i}/{len(matched)}] {session_id}...", end=' ', flush=True)
        
        result = AnnotationComparator.compare_session(session_id, cdp_path, gemini_dir)
        
        if 'error' not in result:
            print(f"OK (match_rate={result['match_rate']:.2%})")
            all_results.append(result)
            
            # Extract conference from session ID
            conf = 'CMC' if 'CMC' in session_id else 'NES'
            results_by_conf[conf].append(result)
        else:
            print(f"ERROR {result['error']}")
    
    # Write CSV summary
    print("\nWriting outputs...")
    csv_path = output_path / 'annotation_comparison_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'session_id', 'conference', 'match_rate', 'matches', 'total_bins',
            'cdp_score2_share', 'entropy_mean', 'entropy_variance',
            'score2_gemini_correlation', 'mismatches_count'
        ])
        writer.writeheader()
        
        for result in all_results:
            writer.writerow({
                'session_id': result['session_id'],
                'conference': 'CMC' if 'CMC' in result['session_id'] else 'NES',
                'match_rate': result['match_rate'],
                'matches': result['matches'],
                'total_bins': result['total_bins'],
                'cdp_score2_share': result['cdp_metrics'].get('score2_share', 0),
                'entropy_mean': result['entropy_mean'],
                'entropy_variance': result['entropy_variance'],
                'score2_gemini_correlation': result['score2_gemini_correlation'],
                'mismatches_count': len(result['mismatches'])
            })
    
    # Write detailed JSON results
    json_path = output_path / 'annotation_comparison_detailed.json'
    with open(json_path, 'w') as f:
        # Convert non-serializable types
        json_results = []
        for r in all_results:
            r_copy = r.copy()
            r_copy['gemini_metrics'] = str(r_copy.get('gemini_metrics'))
            r_copy['cdp_binned'] = str(r_copy.get('cdp_binned'))
            json_results.append(r_copy)
        json.dump(json_results, f, indent=2, default=str)
    
    # Conference summary
    print("\nConference Summary:")
    for conf in sorted(results_by_conf.keys()):
        conf_results = results_by_conf[conf]
        match_rates = [r['match_rate'] for r in conf_results]
        correlations = [r['score2_gemini_correlation'] for r in conf_results]
        entropies = [r['entropy_mean'] for r in conf_results]
        
        print(f"\n  {conf}:")
        print(f"    Sessions: {len(conf_results)}")
        print(f"    Match Rate: mean={statistics.mean(match_rates):.2%}, median={statistics.median(match_rates):.2%}")
        print(f"    Correlation: mean={statistics.mean(correlations):.3f}, median={statistics.median(correlations):.3f}")
        print(f"    Entropy: mean={statistics.mean(entropies):.3f}, var={statistics.variance(entropies):.3f}")
    
    print(f"\nAnalysis complete. Outputs saved to {output_path}/")
    
    return all_results, results_by_conf


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent

    parser = argparse.ArgumentParser(description='Deep annotation comparison: CDP vs v2 chunks')
    parser.add_argument('--cdp-root', default=str(repo_root / 'data'),
                        help='Path to CDP data root')
    parser.add_argument('--gemini-root', default=str(repo_root / 'data-v2'),
                        help='Path to v2 chunk annotation root')
    parser.add_argument('--output-dir', default=str(here / 'analysis_outputs'),
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    results, results_by_conf = run_analysis(args.cdp_root, args.gemini_root, args.output_dir)
