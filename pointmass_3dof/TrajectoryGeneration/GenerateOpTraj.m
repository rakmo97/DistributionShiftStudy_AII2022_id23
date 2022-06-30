%% Omkar S. Mulekar
% Optimal Trajectory Generator for 3DOF rigid body Lunar Landings
% - Given specified upper and lower bounds for randomized initial
% conditions, this script will generate a specified number (nTrajs) of
% thrust optimal trajectories, i.e. trajectories that minimize the
% magnitude of thrust integrated over time.


clear all
close all 
clc
% %%

run ../../../3DoF_RigidBody/OpenOCL/ocl.m 

%% Generation settings
nTrajs = 15000; % Number of trajectories to generate
plotting = 0; % Plot things or no?
saveout = ['d',datestr(now,'yyyymmdd_HHoMMSSFFF'),'_genTrajs','.mat'];

% Lower and upper values for random initial conditions
% [x,y,z,dx,dy,dz,m]
lower = [-5, -5, -20, -1, -1, -1, 1]';
upper = [ 5,  5,  20,  1,  1,  1, 1]';

% Target State [x,y,z,dx,dy,dz]
target = [0,0,0.1,0,0,-0.1];

% Preallocations
N = 100;
surfFunctionOut = cell(nTrajs,1);
objectiveOut = zeros(nTrajs,2);
Jout = zeros(nTrajs,1);
stateOut = zeros(N,8,nTrajs);
ctrlOut = zeros(N,3,nTrajs);
runTimeOut = zeros(nTrajs,1);
stateFinal = zeros(nTrajs,7);

for i = 1:nTrajs
 
    % Solve OCP
    whilecount = 0;
    err_count = 0;
    while whilecount == err_count
        try   
    

            %% parameters

            conf = struct;
            conf.g = 9.81; % m/s2
            conf.g0 = 9.81; % m/s2
            conf.r = 1;
            conf.Isp = 300;



            %% Setup Solver
            varsfun    = @(sh) landervarsfun(sh, conf);
            daefun     = @(sh,x,z,u,p) landereqfun(sh,x,z,u,conf);
            pathcosts  = @(ch,x,~,u,~) landerpathcosts(ch,x,u,conf);

            solver = ocl.Problem([], varsfun, daefun, pathcosts, 'N',N);

            %% Populate Solver Settings
            % Parameters
            solver.setParameter('g'    , conf.g    );
            solver.setParameter('g0'   , conf.g0   );
            solver.setParameter('r'    , conf.r    );

            % Generate Initial Conditions and Target [x,y,phi,dx,dy,dphi,m]
            r = lower+(upper-lower).*rand(7,1);

            % Set Initial Conditions
            solver.setInitialBounds( 'x'   ,   r(1)   );
            solver.setInitialBounds( 'y'   ,   r(2)   );
            solver.setInitialBounds( 'z' ,   r(3)   );
            solver.setInitialBounds( 'dx'  ,   r(4)   );
            solver.setInitialBounds( 'dy'  ,   r(5)   );
            solver.setInitialBounds( 'dz',    r(6)   );
            solver.setInitialBounds( 'm'   ,   r(7)   );


            % Set Target State
            solver.setEndBounds( 'x' ,    target(1) );
            solver.setEndBounds( 'y' ,    target(2) );
            solver.setEndBounds( 'z' ,  target(3) );
            solver.setEndBounds( 'dx'  ,  target(4) );
            solver.setEndBounds( 'dy'  ,  target(5) );
            solver.setEndBounds( 'dz',  target(6) );



    
    
            %% Run Solver
            disp(['Starting trajectory ',num2str(i),' of ',num2str(nTrajs),'....'])

            tic
            initialGuess    = solver.getInitialGuess();
            [solution,times] = solver.solve(initialGuess);
            timeToRun = toc;
            
            if ~strcmp(solver.solver.stats.return_status,'Solve_Succeeded')
                error('Optimal Solution Not Found, Retrying...')
            end
            
            %% Process solutions
            % Grab Times
            ts = times.states.value;
            tc = times.controls.value;
            ratio = (length(ts)-1)/length(tc); % Ratio of ctrl times to state times

            % Pull out states
            x     = solution.states.x.value;
            y     = solution.states.y.value;
            z   = solution.states.z.value;
            dx    = solution.states.dx.value;
            dy    = solution.states.dy.value;
            dz  = solution.states.dz.value;
            m  = solution.states.m.value;

            xa     = solution.states.x.value;
            ya     = solution.states.y.value;
            za   = solution.states.z.value;
            dxa    = solution.states.dx.value;
            dya    = solution.states.dy.value;
            dza  = solution.states.dz.value;
            ma  = solution.states.m.value;


            % Pull out controls
            Fx = solution.controls.Fx.value;
            Fy = solution.controls.Fy.value;
            Fz = solution.controls.Fz.value;


            % Define indexes of states that align with control values
            idxs = 1:ratio:length(ts)-1;

            % Separate states by available controls
            x     = x(idxs);
            y     = y(idxs);
            z   = z(idxs);
            dx    = dx(idxs);
            dy    = dy(idxs);
            dz  = dz(idxs);
            m     = m(idxs);


            % Calculate Costs
            L_F = Fx.^2 + Fy.^2 + Fz.^2;
            J_F = trapz(tc,L_F);

            J_path = J_F;

            disp(['Path Cost is  ', num2str(J_path)])

            % Save off outputs
            Jout(i,:) = [J_path];
            stateOut(:,:,i) = [tc',x',y',z',dx',dy',dz',m'];
            ctrlOut(:,:,i) = [Fx',Fy',Fz'];
            runTimeOut(i) = timeToRun;
            stateFinal(i,:) = [xa(end),ya(end),za(end),dxa(end),dya(end),dza(end),ma(end)];
            
            if plotting
                % Plot x,y,z trajectory
                figure(1);
                plot3(x(1),y(1),z(1),'rx','MarkerSize',10)
                hold on
                grid on
                plot3(solution.states.x.value,...
                   solution.states.y.value,...
                   solution.states.z.value,...
                   'Color','b','LineWidth',1.5);
                plot3(xa(end),ya(end),za(end),'bo','MarkerSize',10)
                xlabel('x[m]');ylabel('y[m]');; zlabel('z [m]')
                legend('Starting Point','Trajectory','Ending Point','Objective','Surface','location','best')


                % Plot thrust profiles
                figure(2);
                subplot(3,1,1)
                plot(tc,Fx,'g')
                hold on
                title('Controls')
                ylabel('F_x [N]')
                subplot(3,1,2)
                plot(tc,Fy,'b')
                hold on
                ylabel('F_y [N]')                
                subplot(3,1,3)
                plot(tc,Fz,'b')
                hold on
                ylabel('F_z [N]')


                figure(3);
                subplot(2,2,1)
                hold on
                plot(tc,x,'g')
                title('Position vs Time')
                ylabel('x [m]')
                subplot(2,2,2)
                plot(tc,y,'b')
                hold on
                ylabel('y [m]')
                subplot(2,2,3)
                plot(tc,z,'b')
                hold on
                ylabel('z [m]')
                subplot(2,2,4)
                plot(tc,m,'b')
                hold on
                ylabel('m [kg]')

            end
           
        catch
            disp('Optimal Solution Not Found, Retrying...');
            err_count = err_count+1;
        end
        whilecount = whilecount+1;
    end


    disp(['Finished trajectory ',num2str(i),' of ',num2str(nTrajs)])
    
    
    
end
%%
fprintf("\n\nTrajectory Generation Complete!\nSaving Variables to .mat file...\n")
disp(['Filename: ',saveout])
save(saveout,'surfFunctionOut','objectiveOut','Jout','stateOut','ctrlOut','runTimeOut','stateFinal');
fprintf("\nProgram Complete!\n")
disp(['at ',datestr(now,'yyyymmdd_HHoMMSSFFF')])

%% Solver Functions
function landervarsfun(sh, c)


    % Define States
    sh.addState('x');
    sh.addState('y');
    sh.addState('z');
    sh.addState('dx');
    sh.addState('dy');
    sh.addState('dz');
    sh.addState('m');

    Fmax = 20;
    % Define Controls
    sh.addControl('Fx', 'lb', -Fmax, 'ub', Fmax);  % Force [N]
    sh.addControl('Fy', 'lb', -Fmax, 'ub', Fmax);  % Force [N]
    sh.addControl('Fz', 'lb', 0, 'ub', Fmax);  % Force [N]


    % System Parameters
    sh.addParameter('g');
    sh.addParameter('g0');
    sh.addParameter('r');
    sh.addParameter('Isp')
    sh.addParameter('objective');
    sh.addParameter('surfFunc');

end

function landereqfun(sh,x,~,u,c) % https://charlestytler.com/quadcopter-equations-motion/


    

    
    % Equations of Motion
    sh.setODE( 'x'   , x.dx);
    sh.setODE( 'y'   , x.dy);
    sh.setODE( 'z'   , x.dz);
    sh.setODE( 'dx'  , (1/x.m)*u.Fx);
    sh.setODE( 'dy'  , (1/x.m)*u.Fy);
    sh.setODE( 'dz'  , (1/x.m)*u.Fz - c.g);
    sh.setODE( 'm'   , -sqrt((u.Fx)^2 + (u.Fy)^2 + (u.Fz)^2) / (c.Isp*c.g0));


end

function landerpathcosts(ch,x,u,~)
    
    % Cost Function (thrust magnitude)
    ch.add(u.Fx^2);
    ch.add(u.Fy^2);
    ch.add(u.Fz^2);


end



