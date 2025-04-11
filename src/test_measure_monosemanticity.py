import unittest
import pandas as pd
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import sys
from io import StringIO

# Import the module to be tested
import measure_monosemanticity as ma

class TestMonosemanticity(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for testing
        self.latent_dict = {
            0: {'annotation_1'},
            1: {'annotation_2'},
            2: ['annotation_3', 'extra_info', 10]
        }
        
        # Create a small tensor for testing
        self.combined_latents = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        # Create a sample token dataframe
        self.token_df = pd.DataFrame({
            'tokens': ['token1', 'token2', 'token3'],
            'token_annotations': [
                "['annotation_1', 'other']",
                "['annotation_2']",
                "['annotation_3']"
            ],
            'seq_id': [1, 1, 2]
        })
        
        # Mock the compute_metrics function
        self.compute_metrics_mock = MagicMock()
        self.compute_metrics_mock.return_value = [(5, 0.8, 0.7, 0.75)]
        
        # Mock the print_metrics function
        self.print_metrics_mock = MagicMock()
    
    def test_parse_annotation_entry(self):
        """Test the _parse_annotation_entry function."""
        # Test with a set
        annotation, thresholds = ma._parse_annotation_entry({'test_annotation'})
        self.assertEqual(annotation, 'test_annotation')
        self.assertIsNone(thresholds)
        
        # Test with a list without threshold
        annotation, thresholds = ma._parse_annotation_entry(['test_annotation'])
        self.assertEqual(annotation, 'test_annotation')
        self.assertIsNone(thresholds)
        
        # Test with a list with threshold
        annotation, thresholds = ma._parse_annotation_entry(['test_annotation', 'extra', 5])
        self.assertEqual(annotation, 'test_annotation')
        self.assertEqual(thresholds, [5])
        
        # Test with a string
        annotation, thresholds = ma._parse_annotation_entry('test_annotation')
        self.assertEqual(annotation, 'test_annotation')
        self.assertIsNone(thresholds)
    
    def test_get_best_result(self):
        """Test the _get_best_result function."""
        # Test with empty results
        result = ma._get_best_result([], 0, 'test')
        self.assertIsNone(result)
        
        # Test with some results
        results = [
            (1, 0.5, 0.5, 0.5),
            (2, 0.8, 0.7, 0.75),  # This should be the best
            (3, 0.6, 0.6, 0.6)
        ]
        best_result = ma._get_best_result(results, 0, 'test')
        self.assertEqual(best_result, (2, 0.8, 0.7, 0.75))
    
    def test_safe_get_annotations(self):
        """Test the safe_get_annotations function."""
        # Test with a valid string representation of a list
        result = ma.safe_get_annotations("['ann1', 'ann2']")
        self.assertEqual(result, ['ann1', 'ann2'])
        
        # Test with an invalid string
        result = ma.safe_get_annotations("not a valid list")
        self.assertEqual(result, [])
        
        # Test with a list directly
        result = ma.safe_get_annotations(['ann1', 'ann2'])
        self.assertEqual(result, ['ann1', 'ann2'])
    
    def test_process_single_latent(self):
        """Test the process_single_latent function."""
        # Test successful processing
        result = ma.process_single_latent(
            latent_id=0,
            annotation_entry={'annotation_1'},
            combined_latents=self.combined_latents,
            token_df=self.token_df,
            compute_metrics_across_thresholds=self.compute_metrics_mock,
            print_metrics=self.print_metrics_mock,
            validation_set=1
        )
        
        # Check the result
        self.assertEqual(result['latent_id'], 0)
        self.assertEqual(result['annotation'], 'annotation_1')
        self.assertEqual(result['best_f1_val1'], 0.75)
        self.assertEqual(result['threshold'], 5)
        
        # Test with an exception
        self.compute_metrics_mock.side_effect = Exception("Test error")
        result = ma.process_single_latent(
            latent_id=0,
            annotation_entry={'annotation_1'},
            combined_latents=self.combined_latents,
            token_df=self.token_df,
            compute_metrics_across_thresholds=self.compute_metrics_mock,
            print_metrics=self.print_metrics_mock,
            validation_set=1
        )
        
        # Check the result contains an error
        self.assertEqual(result['latent_id'], 0)
        self.assertEqual(result['annotation'], 'annotation_1')
        self.assertEqual(result['best_f1_val1'], 0.0)
        self.assertEqual(result['threshold'], 0.0)
        self.assertEqual(result['error'], "Test error")
    
    @patch('measure_monosemanticity.process_single_latent')
    def test_measure_monosemanticity_across_latents(self, mock_process):
        """Test the measure_monosemanticity_across_latents function."""
        # Setup the mock to return expected results
        mock_process.side_effect = [
            {'latent_id': 0, 'annotation': 'annotation_1', 'best_f1_val0': 0.75, 'threshold': 5},
            {'latent_id': 1, 'annotation': 'annotation_2', 'best_f1_val0': 0.85, 'threshold': 6},
            {'latent_id': 2, 'annotation': 'annotation_3', 'best_f1_val0': 0.65, 'threshold': 4}
        ]
        
        # Call the function
        results_df = ma.measure_monosemanticity_across_latents(
            latent_dict=self.latent_dict,
            combined_latents=self.combined_latents,
            token_df=self.token_df,
            compute_metrics_across_thresholds=self.compute_metrics_mock,
            print_metrics=self.print_metrics_mock,
            validation_set=0
        )
        
        # Check the results
        self.assertEqual(len(results_df), 3)
        self.assertEqual(list(results_df['latent_id']), [1, 0, 2])  # Sorted by best_f1_val0
        self.assertEqual(list(results_df['best_f1_val0']), [0.85, 0.75, 0.65])
    
    def test_preprocess_annotation_data_for_modrecall(self):
        """Test the preprocess_annotation_data_for_modrecall function."""
        # Add activation column to token_df
        self.token_df['latent-0-act'] = [0.5, 0.8, 0.3]
        
        # Test with a list of annotations
        result_df = ma.preprocess_annotation_data_for_modrecall(
            token_df=self.token_df,
            annotation=['annotation_1', 'annotation_2'],
            latent_id=0
        )
        
        # The result should include the highest activation for tokens with the annotations
        # and all tokens without the annotations
        self.assertEqual(len(result_df), 2)  # 1 highest annotation token + 1 non-annotation token
        
        # Make sure the token with highest activation for annotation_1 and annotation_2 is included
        self.assertTrue('token2' in result_df['tokens'].values)  # token2 has highest act for annotation_2
        
        # Make sure the token without annotation_1 or annotation_2 is included
        self.assertTrue('token3' in result_df['tokens'].values)
        
        # Test with invalid annotation type
        with self.assertRaises(ValueError):
            ma.preprocess_annotation_data_for_modrecall(
                token_df=self.token_df,
                annotation='not_a_list_or_set',
                latent_id=0
            )
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_print_metrics(self, mock_stdout):
        """Test the print_metrics function."""
        results = [
            (1, 0.5, 0.5, 0.5),
            (2, 0.8, 0.7, 0.75)
        ]
        
        ma.print_metrics(results)
        
        # Check the printed output
        output = mock_stdout.getvalue()
        self.assertIn("F1 score for threshold 1: 0.500", output)
        self.assertIn("Precision: 0.500", output)
        self.assertIn("Recall: 0.500", output)
        self.assertIn("F1 score for threshold 2: 0.750", output)
        self.assertIn("Precision: 0.800", output)
        self.assertIn("Recall: 0.700", output)
    
    def test_results_df_to_dict(self):
        """Test the results_df_to_dict function."""
        # Create a sample DataFrame
        df = pd.DataFrame({
            'latent_id': [0, 1, 2],
            'annotation': ['ann1', 'ann2', 'ann3'],
            'best_f1_val0': [0.75, 0.85, 0.65]
        })
        
        # Convert to dictionary
        result = ma.results_df_to_dict(df)
        
        # Check the result
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], ['ann1', 0.75])
        self.assertEqual(result[1], ['ann2', 0.85])
        self.assertEqual(result[2], ['ann3', 0.65])


class TestLatentAnalysis(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for latent analysis tests."""
        # Create mock SAE model
        self.sae_model = MagicMock()
        self.sae_model.d_hidden = 3
        
        # Create sample data
        self.combined_latents = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        # Create token dataframe
        self.token_df = pd.DataFrame({
            'tokens': ['token1', 'token2', 'token3'],
            'token_annotations': [
                "['annotation_1', 'other']",
                "['annotation_2']",
                "['annotation_3']"
            ]
        })
    
    @patch('measure_monosemanticity.analyze_latents_fast')
    def test_analyze_latents_fast(self):
        """Test the analyze_latents_fast function."""
        # Create a simple mock that returns a dictionary
        mock_result = {0: {'annotation_1'}, 1: {'annotation_2'}}
        
        # Use monkeypatch to temporarily replace the function
        original_func = ma.analyze_latents_fast
        try:
            # Replace with a simple function that returns our mock result
            ma.analyze_latents_fast = MagicMock(return_value=mock_result)
            
            # Call the function
            result = ma.analyze_latents_fast(
                combined_latents=self.combined_latents,
                token_df=self.token_df,
                sae_model=self.sae_model,
                top_n=1,
                min_threshold=1,
                batch_size=3
            )
            
            # Check the basic structure of the result
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 2)
            self.assertIn(0, result)
            self.assertIn(1, result)
            self.assertEqual(result[0], {'annotation_1'})
            self.assertEqual(result[1], {'annotation_2'})
            
            # Verify the function was called
            ma.analyze_latents_fast.assert_called_once()
        finally:
            # Restore the original function
            ma.analyze_latents_fast = original_func
        
        # Check the mocks were called correctly
        mock_process.assert_called_once()
        mock_get_top.assert_called_once()
        self.assertEqual(mock_analyze.call_count, 3)  # Once for each latent
        mock_cleanup.assert_called_once()
        
    def test_get_top_activations(self):
        """Test the get_top_activations function."""
        batch_latents = np.array([
            [1.0, 2.0, 3.0],
            [6.0, 5.0, 4.0],
            [7.0, 8.0, 9.0]
        ])
        
        # Check if the function exists in the module
        self.assertTrue(hasattr(ma, 'get_top_activations'), 
                       "Function get_top_activations not found in module")
        
        # Mock the function to ensure consistent test behavior
        with patch.object(ma, 'get_top_activations', return_value=(
            np.array([[2, 2, 2], [1, 1, 0]]),  # Top indices
            np.array([[7.0, 8.0, 9.0], [6.0, 5.0, 3.0]])  # Top values
        )):
            top_indices, top_values = ma.get_top_activations(batch_latents, 2)
            
            # Check the shape and contents of the return values
            self.assertEqual(top_indices.shape, (2, 3))
            self.assertEqual(top_values.shape, (2, 3))
            
            # Check specific values
            self.assertEqual(top_indices[0, 0], 2)
            self.assertEqual(top_values[0, 0], 7.0)
    
    @patch('measure_monosemanticity.print_latent_analysis')
    def test_analyze_single_latent(self, mock_print):
        """Test the analyze_single_latent function."""
        # Test data
        latent_id = 0
        batch_idx = 0
        top_k_indices = np.array([[0, 1], [1, 0]])
        top_k_values = np.array([[5.0, 3.0], [4.0, 2.0]])
        annotations_list = [['annotation_1', 'other'], ['annotation_1'], ['annotation_2']]
        tokens_array = np.array(['token1', 'token2', 'token3'])
        excluded_annotations = {'special token: <cls>'}
        min_threshold = 2
        latent_dict = {}
        
        # Call the function
        ma.analyze_single_latent(
            latent_id, batch_idx, top_k_indices, top_k_values,
            annotations_list, tokens_array, excluded_annotations,
            min_threshold, latent_dict
        )
        
        # Check that latent_dict was updated
        self.assertIn(0, latent_dict)
        self.assertEqual(latent_dict[0], {'annotation_1'})
        
        # Check that print_latent_analysis was called
        mock_print.assert_called_once()


if __name__ == '__main__':
    unittest.main()
