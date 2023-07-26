using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public unsafe struct Voxel{

    public fixed float color[3];
    // public fixed float velocity_buffer[6];
    // public fixed float velocity_b[3];
    // public fixed float pressure_buffer[6];
    public fixed float density_buffer[2];
    // public fixed float velocity_sources_buffer[3];
    public int obstacle;
    // public int density_source;
};


public class fluid_simulator : MonoBehaviour
{
    public int N = 4;
    public Mesh mesh;
    public Material material;
    public ComputeShader fluid_shader;
    public ComputeBuffer fluid_buffer;
    private int number_of_voxels;

    private int black_checkerboard_kernel_id, red_checkerboard_kernel_id, density_diffusion_kernel_id, swap_density_kernel_id;

    private List<GameObject> objects;

    private Voxel [] voxels;
    // Start is called before the first frame update


    private void create_voxel(int x, int y){
        GameObject cube = new GameObject("Cube " + x * N + y, typeof(MeshFilter), typeof(MeshRenderer));
        cube.GetComponent<MeshFilter>().mesh = mesh;
        cube.GetComponent<MeshRenderer>().material = new Material(material);
        cube.transform.position = new Vector3(x, y, Random.Range(-.1f, -.1f));

        Color color = new Color(0, 0, 0);
        cube.GetComponent<MeshRenderer>().material.SetColor("_Color", color);

        objects.Add(cube);

        Voxel cube_data = new Voxel();

        int index = x * N + y;
        voxels[index] = cube_data;

        int max = N - 1;

        //edge case
        if(x == 0 || x == max || y == 0 || y == max){
            voxels[index].obstacle = 1;
        }
        else{
            unsafe
            {
                float random_density_value = Random.Range(0,.2f);
                voxels[index].density_buffer[0] = random_density_value;
                voxels[index].density_buffer[1] = random_density_value;
            }
        }

    }

    public void create_cubes(){
        objects = new List<GameObject>();
        voxels = new Voxel[N * N];
        for(int y = 0; y < N; y++){
            for(int x = 0; x < N; x ++){
                create_voxel(x, y);
            }
        }
    }

    void create_shader(){
        int size_in_bytes = 0;

        unsafe
        {
            size_in_bytes=sizeof(Voxel);
        }

        fluid_buffer = new ComputeBuffer(voxels.Length, size_in_bytes);
    }

    void init_iteration_parameters(){

    }

    void dispatch_checkerboard(){
        
        for(int i = 0; i < 1; i++){
            fluid_shader.Dispatch(black_checkerboard_kernel_id, voxels.Length/64, 1, 1);
            fluid_shader.Dispatch(red_checkerboard_kernel_id, voxels.Length/64, 1, 1);
            fluid_shader.Dispatch(swap_density_kernel_id, voxels.Length/16, 1, 1);
        }
    }

    private void read_colors_from_gpu(){
        fluid_buffer.GetData(voxels);
        for (int i = 0; i < objects.Count; i++){
            GameObject obj = objects[i];
            Voxel voxel = voxels[i];
            float[] voxel_float_array;
            unsafe
            {
                voxel_float_array =  new[]{ voxel.color[0], voxel.color[1], voxel.color[2]};
            }
            Color voxel_color = new Color(voxel_float_array[0], voxel_float_array[1], voxel_float_array[2]);
            print(voxel_color);
            obj.GetComponent<MeshRenderer>().material.SetColor("_Color", voxel_color);
        }
    }

    void Start()
    {
        create_cubes();
        create_shader();
        
        fluid_shader.SetInt("N", N);
        black_checkerboard_kernel_id = fluid_shader.FindKernel("checkerboard_black");
        red_checkerboard_kernel_id = fluid_shader.FindKernel("checkerboard_red");
        swap_density_kernel_id = fluid_shader.FindKernel("swap_density");

        fluid_shader.SetBuffer(black_checkerboard_kernel_id, "voxels", fluid_buffer);
        fluid_shader.SetBuffer(red_checkerboard_kernel_id, "voxels", fluid_buffer);
        fluid_shader.SetBuffer(swap_density_kernel_id, "voxels", fluid_buffer);

        //density_diffusion_kernel_id = fluid_shader.FindKernel("density_diffusion");
        init_iteration_parameters();
        
    }



    // Update is called once per frame
    void Update()
    {
        //keep the center pixel white
        unsafe
        {
            int ic = (N*N)/2 + N/2;
            float random_density_value = Random.Range(0,.8f);
            voxels[ic].density_buffer[0] = 1;
            voxels[ic].density_buffer[1] = 1;
        }
        fluid_buffer.SetData(voxels);
        dispatch_checkerboard();
        read_colors_from_gpu();
    }
}
