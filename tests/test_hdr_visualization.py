import tempfile
import unittest
from pathlib import Path
import numpy as np
from utils.hdr_visualization import fit_reinhard,read_hdr,save_preview,tonemap_hdr,write_exr
from utils.project_trajectory import project_page_cameras


class HDRVisualizationTests(unittest.TestCase):
    def test_black_frame_and_extreme_radiance_are_displayable(self):
        image=np.zeros((8,8,3),dtype=np.float32);image[0,0]=[10000,2000,100]
        settings=fit_reinhard([image],stride=1)
        result=tonemap_hdr(image,settings)
        self.assertEqual(result.dtype,np.uint8)
        self.assertTrue((result[1:]==0).all())
        self.assertGreater(result[0,0].max(),0)
        self.assertFalse(tonemap_hdr(np.zeros_like(image)).any())

    def test_fixed_calibration_does_not_pump_exposure_between_frames(self):
        first=np.full((16,16,3),.2,dtype=np.float32);second=first.copy();second[0,0]=100
        settings=fit_reinhard([first,second],stride=1)
        np.testing.assert_array_equal(tonemap_hdr(first,settings)[1:],tonemap_hdr(second,settings)[1:])
        before=first.copy();tonemap_hdr(first,settings);np.testing.assert_array_equal(first,before)

    def test_exposure_increases_monotonically(self):
        values=np.logspace(-4,3,80,dtype=np.float32);image=np.repeat(values[None,:,None],3,axis=2)
        settings=fit_reinhard([image],stride=1)
        a=tonemap_hdr(image,settings,-1);b=tonemap_hdr(image,settings,0);c=tonemap_hdr(image,settings,1)
        self.assertTrue((a<=b).all() and (b<=c).all())
        self.assertTrue((np.diff(b[0,:,0].astype(int))>=0).all())

    def test_float_exr_roundtrip_preserves_radiance_and_rgb_order(self):
        image=np.array([[[.01,.3,7.5],[1024,2,.125]],[[0,0,0],[3,4,5]]],dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'raw.exr';write_exr(path,image);np.testing.assert_array_equal(read_hdr(path),image)
            output=Path(directory)/'preview.png';save_preview(output,image);self.assertTrue(output.is_file())

    def test_invalid_hdr_values_are_rejected(self):
        for value in [np.nan,np.inf]:
            image=np.ones((2,2,3),dtype=np.float32);image[0,0,0]=value
            with self.assertRaises(ValueError):tonemap_hdr(image)

    def test_camera_winding_is_continuous_through_axial_reversal(self):
        preset=dict(fps=60,duration_seconds=16,turn_blend_seconds=.6,turns_per_leg=2,
                    lateral_radius_x=1.,lateral_radius_y=.7,forward_travel=3.,camera_to_world_start=np.eye(4).tolist())
        with self.assertRaises(ValueError):
            project_page_cameras(preset, fps=0)
        path,times=project_page_cameras(preset);centers=path[:,:3,3]
        angles=np.unwrap(np.arctan2(centers[:,1]/.7,centers[:,0]+1))
        self.assertTrue((np.diff(angles)>0).all())
        self.assertTrue((np.diff(centers[:481,2])>0).all() and (np.diff(centers[480:,2])<0).all())
        self.assertEqual(len(path),960);self.assertAlmostEqual(times[-1],16-1/60)
        steps=np.roll(centers,-1,axis=0)-centers
        self.assertGreater(np.linalg.norm(steps,axis=1).min(),0)
        self.assertLess(np.linalg.norm(steps[-1]-steps[0]),.002)


if __name__=='__main__':unittest.main()
