vertex_shader =\
'''
#version 130

uniform mat4 normalizeMat;

varying vec3 normal;
varying vec3 fragPos;
varying vec4 pos;

void main() {
    gl_Position = gl_ModelViewProjectionMatrix * normalizeMat * gl_Vertex;
    fragPos = (gl_ModelViewMatrix * gl_Vertex).xyz;
    normal = gl_Color.xyz;
    pos = gl_Vertex;
}
'''

def fragment_shader(texturesCount,drawFrameIndex):
     
    return '''
    #version 130

    vec3 lightColor = vec3(0.3,0.3,0.3);

    // ambient light
    float ambientStrength = 1.0;

    // diffuse light
    varying vec3 normal;
    varying vec3 fragPos;
    vec3 lightPos=vec3(0.5,0.5,0.5);

    // specular light
    uniform vec3 viewPos;
    float specularStrength = 0.5;

    uniform sampler2D depthMap[{0}];
    uniform sampler2D projectTex[{0}];

    varying vec4 pos;
    uniform vec3 cameraPose[{0}];
    uniform mat4 projectMat[{0}];

    uniform float wstep;
    uniform float hstep;

    vec3 lightingColor(vec3 color){{
        // ambient light
        vec3 ambient = ambientStrength * lightColor;

        // diffuse light
        vec3 norm = normalize(normal);
        vec3 lightDir = normalize(lightPos - fragPos);  
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;

        // specular light
        vec3 viewDir = normalize(viewPos - fragPos);
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * lightColor;        

        vec3 result = (ambient + diffuse + specular) * color;

        return result;
    }}

    vec4 projectColor(sampler2D tex,sampler2D depthMap,mat4 mat){{
        vec4 p = mat * pos;
        p /= p.w;
        vec2 uv = p.xy*0.5+0.5;
        
        // shrink boundary
        float isBoundary = 1;
        int kernelsize=3;
        for(int i=-(kernelsize-1)/2;i<=(kernelsize-1)/2;i++){{
            float isZero = 0;
            for(int j=-(kernelsize-1)/2;j<=(kernelsize-1)/2;j++){{
                vec4 color = texture2D(tex,uv+vec2(wstep*i,hstep*j));
                if(color.x<1e-1&&color.y<1e-1&&color.z<1e-1){{
                    isZero = 1;
                    isBoundary = 0;
                    break;
                }}
            }}    
            if (isZero==1){{
                break;
            }}
        }}
        if (isBoundary==0){{
            return vec4(0.0,0.0,0.0,0.0);
        }}
        
        vec4 color = vec4(0.0,0.0,0.0,0.0);
        float depthValue = texture2D(depthMap,uv).x;
        if ((p.z-depthValue)<3e-3){{
            color = texture2D(tex,uv);
        }}
        return color;
    }}

    vec4 blendviewColors(){{
        int validCount=0;
        vec4 finalColor = vec4(0.0,0.0,0.0,0.0);
        for(int i=0;i<{0};i++){{
            vec4 color = projectColor(projectTex[i],depthMap[i],projectMat[i]);
            if(color.a!=0){{
                
                if({1} == 1 ){{
                    color = vec4(float(i)/float({0}),0.0,0.0,1.0);
                }}

                finalColor += color;
                validCount++;
            }}
        }}
        return vec4(finalColor.xyz/validCount,finalColor.a);        
    }}

    vec4 getClosetCamera(){{
        int closestCameraIndex = -1;
        float minDistance = 1e+100;
        vec4 color = vec4(0.0,0.0,0.0,1.0);

        for(int i=0;i<{0};i++){{
            vec4 c = projectColor(projectTex[i],depthMap[i],projectMat[i]);
            if(c.a==0){{
                continue;
            }}
            vec3 cam2point = cameraPose[i]-pos.xyz;
            float angle = acos( dot( normalize(normal), normalize(cam2point) ) );

            if(angle<minDistance){{
                minDistance = angle;
                closestCameraIndex = i;
                color = c;
            }}
        }}
        if({1} == 1 ){{
            color = vec4(float(closestCameraIndex)/float({0}),0.0,0.0,1.0);
        }}
        return color;
    }}

    void main() {{
        int i = 0;
        gl_FragColor = getClosetCamera();
    }}'''.format(texturesCount,drawFrameIndex)