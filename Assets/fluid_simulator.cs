using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public unsafe struct Voxel{

    public fixed float color[3];
    public fixed float velocity_buffer[6];
    public float divergence;
    public float pressure;
    // public fixed float velocity_b[3];
    // public fixed float pressure_buffer[6];
    public fixed float density_buffer[2];
    // public fixed float velocity_sources_buffer[3];
    public int obstacle;
    public float density_source;
};

public class fluid_simulator : MonoBehaviour
{
    public int N = 4;
    public float diffusion;
    public float dt;
    public Mesh mesh;
    public Material material;
    public ComputeShader fluid_shader;
    public ComputeBuffer fluid_buffer;
    public RenderTexture render_texture;
    private int number_of_voxels;

    private int black_checkerboard_kernel_id, red_checkerboard_kernel_id, density_diffusion_kernel_id, swap_density_kernel_id, 
    density_advect_kernel_id, set_color_kernel_id, velocity_advect_kernel_id, swap_velocity_kernel_id, calculate_pressure_black_kernel_id, calculate_pressure_red_kernel_id,
    calculate_divergence_kernel_id, remove_pressure_kernel_id;

    private List<GameObject> objects;

    private Voxel [,] voxels;
    // Start is called before the first frame update


    private void create_voxel(int x, int y){

        voxels[y,x] = new Voxel();

        int max = N - 1;

        //edge case
        if(x == 0 || x == max || y == 0 || y == max){
            voxels[y,x] .obstacle = 1;
        }
        else{
            unsafe
            {
                float value = (float)(.5*x+.5*y)/N;
                print(value + ", " + x + ", " + y + ", " + N);
                voxels[y,x] .density_buffer[0] = value;
                voxels[y,x] .density_buffer[1] = value;//(x+y)/(2*N);
            }
        }

    }

    public void create_cubes(){
        objects = new List<GameObject>();
        voxels = new Voxel[N , N];
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

    void dispatch_diffusion(){
        //diffuse
        for(int i = 0; i < 4; i++){
            fluid_shader.Dispatch(black_checkerboard_kernel_id, voxels.Length/8, voxels.Length/8, 1);
            fluid_shader.Dispatch(red_checkerboard_kernel_id, voxels.Length/8, voxels.Length/8, 1);
            fluid_shader.Dispatch(swap_density_kernel_id, voxels.Length/8, voxels.Length/8, 1);
            fluid_shader.Dispatch(swap_velocity_kernel_id, voxels.Length/8, voxels.Length/8, 1);
        }
    }

    void add_sources(){
        unsafe
        {
            int x_center = N/2;
            int y_center = N/2;
            float random_density_value = Random.Range(0, .8f);
            voxels[y_center, x_center].density_buffer[0] = 1;
            voxels[y_center, x_center].density_buffer[1] = 1;
            voxels[y_center, x_center].density_source = 1;
            voxels[y_center, x_center].obstacle = 0;
        }
    }

    void add_x_wall(int x1, int x2, int y){

        int smaller, bigger;

        if(x1 > x2){
            bigger = x1;
            smaller = x2;
        }
        else{
            bigger = x2;
            smaller = x1;
        }
        for(int i = smaller; i <= bigger; i++){
            voxels[y,i].obstacle = 1;
            unsafe{
                voxels[y,i].density_buffer[0] = 0;
                voxels[y,i].density_buffer[1] = 0;
            }
        }
    }

    void add_y_wall(int y1, int y2, int x){

        int smaller, bigger;

        if(y1 > y2){
            bigger = y1;
            smaller = y2;
        }
        else{
            bigger = y2;
            smaller = y1;
        }
        for(int i = smaller; i <= bigger; i++){
            voxels[i,x].obstacle = 1;
            unsafe{
                voxels[i,x].density_buffer[0] = 0;
                voxels[i,x].density_buffer[1] = 0;
            }
        }
    }

    void Start()
    {
        float a = dt*diffusion*N*N;
        float b = 1/(1+a);
        create_cubes();
        create_shader();
        
        fluid_shader.SetInt("N", N);
        fluid_shader.SetFloat("a", a);
        fluid_shader.SetFloat("b", b);
        fluid_shader.SetFloat("dt", dt);

        black_checkerboard_kernel_id = fluid_shader.FindKernel("checkerboard_black");
        red_checkerboard_kernel_id = fluid_shader.FindKernel("checkerboard_red");

        density_advect_kernel_id = fluid_shader.FindKernel("density_advection");
        velocity_advect_kernel_id = fluid_shader.FindKernel("velocity_advection");
        swap_density_kernel_id = fluid_shader.FindKernel("swap_density");
        swap_velocity_kernel_id = fluid_shader.FindKernel("swap_velocity");
        set_color_kernel_id = fluid_shader.FindKernel("set_color");

        calculate_pressure_black_kernel_id = fluid_shader.FindKernel("calculate_pressure_black");
        calculate_pressure_red_kernel_id = fluid_shader.FindKernel("calculate_pressure_red");
        calculate_divergence_kernel_id = fluid_shader.FindKernel("calculate_divergence");
        remove_pressure_kernel_id = fluid_shader.FindKernel("remove_pressure");

        render_texture = new RenderTexture(N, N, 24);
        render_texture.enableRandomWrite = true;
        render_texture.Create();
        
        fluid_shader.SetTexture(set_color_kernel_id, "Colors", render_texture);

        fluid_shader.SetBuffer(black_checkerboard_kernel_id, "voxels", fluid_buffer);
        fluid_shader.SetBuffer(red_checkerboard_kernel_id, "voxels", fluid_buffer);

        fluid_shader.SetBuffer(density_advect_kernel_id, "voxels", fluid_buffer);
        fluid_shader.SetBuffer(velocity_advect_kernel_id, "voxels", fluid_buffer);

        fluid_shader.SetBuffer(set_color_kernel_id, "voxels", fluid_buffer);
        fluid_shader.SetBuffer(swap_density_kernel_id, "voxels", fluid_buffer);
        fluid_shader.SetBuffer(swap_velocity_kernel_id, "voxels", fluid_buffer);

        fluid_shader.SetBuffer(calculate_pressure_black_kernel_id, "voxels", fluid_buffer);
        fluid_shader.SetBuffer(calculate_pressure_red_kernel_id, "voxels", fluid_buffer);
        fluid_shader.SetBuffer(calculate_divergence_kernel_id, "voxels", fluid_buffer);
        fluid_shader.SetBuffer(remove_pressure_kernel_id, "voxels", fluid_buffer);

        //density_diffusion_kernel_id = fluid_shader.FindKernel("density_diffusion");

        add_x_wall(N/2 - N/4, N/2 + N/4, N/2 - 2);
        add_y_wall(N/2 - N/4, N/2 + N/4, N/2 + 2);
        add_sources();
  

        fluid_buffer.SetData(voxels);


    }



    // Update is called once per frame
    void Update()
    {
        //change sources
        //SWAP

        //diffuse voxels
        dispatch_diffusion();

        //project


        //SWAP
        //fluid_shader.Dispatch(swap_velocity_kernel_id, N/8, N/8, 1);

        //advect
        //fluid_shader.Dispatch(density_advect_kernel_id, N/8, N/8, 1);
        //fluid_shader.Dispatch(velocity_advect_kernel_id, N/8, N/8, 1);

        //project
        
        //set final color
        fluid_shader.Dispatch(set_color_kernel_id, N/8, N/8, 1);
    }

    private void OnRenderImage(RenderTexture src, RenderTexture dest){

        if(render_texture == null){
            render_texture = new RenderTexture(N, N, 24);
            render_texture.enableRandomWrite = true;
            render_texture.Create();
        }
        Graphics.Blit(render_texture, dest);
    }
}
