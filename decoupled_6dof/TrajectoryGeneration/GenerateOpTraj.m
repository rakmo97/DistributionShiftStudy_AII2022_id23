%% Omkar S. Mulekar
% Optimal Trajectory Generator for 6DoF Rigidbody


clear all
close all
clc
% %%

run ../../../3DoF_RigidBody/OpenOCL/ocl.m 

%% Generation settings
nTrajs = 2000; % Number of trajectories to generate
plotting = 1; % Plot things or no?
saveout = ['d',datestr(now,'yyyymmdd_HHoMM'),'_genTrajs','.mat'];

% Lower and upper values for random initial conditions
% [x,y,z,dx,dy,dz,phi,theta,psi,p,q,r, m]
lower = [-5, -5, -20, -1, -1, -1 -pi/10, -pi/10, -pi/10, -1, -1, -1, 1]';
upper = [ 5,  5,  20,  1,  1,  1, pi/10,  pi/10,  pi/10,  1,  1,  1, 1]';

% Target State [x,y,z,dx,dy,dz,phi,theta,psi,p,q,r]
target = [0, 0, 0.1, 0, 0, -0.1, 0, 0, 0, 0, 0, 0];

% Preallocations
N = 100;
surfFunctionOut = cell(nTrajs,1);
objectiveOut = zeros(nTrajs,2);
Jout = zeros(nTrajs,3);
stateOut = zeros(N,14,nTrajs);
ctrlOut = zeros(N,4,nTrajs);
runTimeOut = zeros(nTrajs,1);
stateFinal = zeros(nTrajs,13);

for i = 1:nTrajs
 
%     % Solve OCP
    whilecount = 0;
    err_count = 0;
    while whilecount == err_count
        try   
    

            %% parameters

            conf = struct;
            conf.g = 9.81; % m/s2
            conf.g0 = 9.81; % m/s2
            conf.r = 0.25;
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
            r = lower+(upper-lower).*rand(13,1);

            % Set Initial Conditions
            solver.setInitialBounds( 'x'    ,   r(1)   );
            solver.setInitialBounds( 'y'    ,   r(2)   );
            solver.setInitialBounds( 'z'    ,   r(3)   );
            solver.setInitialBounds( 'dx'   ,   r(4)   );
            solver.setInitialBounds( 'dy'   ,   r(5)   );
            solver.setInitialBounds( 'dz'   ,   r(6)   );
            solver.setInitialBounds( 'phi'  ,   r(7)   );
            solver.setInitialBounds( 'theta',   r(8)   );
            solver.setInitialBounds( 'psi'  ,   r(9)   );
            solver.setInitialBounds( 'p'    ,   r(10)  );
            solver.setInitialBounds( 'q'    ,   r(11)  );
            solver.setInitialBounds( 'r'    ,   r(12)  );
            solver.setInitialBounds( 'm'    ,   r(13)  );


            % Set Target State
            solver.setEndBounds( 'x'     ,    target(1)  );
            solver.setEndBounds( 'y'     ,    target(2)  );
            solver.setEndBounds( 'z'     ,    target(3)  );
            solver.setEndBounds( 'dx'    ,    target(4)  );
            solver.setEndBounds( 'dy'    ,    target(5)  );
            solver.setEndBounds( 'dz'    ,    target(6)  );
            solver.setEndBounds( 'phi'   ,    target(7)  );
            solver.setEndBounds( 'theta' ,    target(8)  );
            solver.setEndBounds( 'psi'   ,    target(9)  );
            solver.setEndBounds( 'p'     ,    target(10) );
            solver.setEndBounds( 'q'     ,    target(11) );
            solver.setEndBounds( 'r'     ,    target(12) );



    
    
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
            x       = solution.states.x.value;
            y       = solution.states.y.value;
            z       = solution.states.z.value;
            dx      = solution.states.dx.value;
            dy      = solution.states.dy.value;
            dz      = solution.states.dz.value;
            phi     = solution.states.phi.value;
            theta   = solution.states.theta.value;
            psi     = solution.states.psi.value;
            p       = solution.states.p.value;
            q       = solution.states.q.value;
            r       = solution.states.r.value;
            m       = solution.states.m.value;

            xa       = solution.states.x.value;
            ya       = solution.states.y.value;
            za       = solution.states.z.value;
            dxa      = solution.states.dx.value;
            dya      = solution.states.dy.value;
            dza      = solution.states.dz.value;
            phia     = solution.states.phi.value;
            thetaa   = solution.states.theta.value;
            psia     = solution.states.psi.value;
            pa       = solution.states.p.value;
            qa       = solution.states.q.value;
            ra       = solution.states.r.value;
            ma       = solution.states.m.value;


            % Pull out controls
            u1 = solution.controls.u1.value;
            u2 = solution.controls.u2.value;
            u3 = solution.controls.u3.value;
            u4 = solution.controls.u4.value;


            % Define indexes of states that align with control values
            idxs = 1:ratio:length(ts)-1;

            % Separate states by available controls
            x       = x(idxs);
            y       = y(idxs);
            z       = z(idxs);
            dx      = dx(idxs);
            dy      = dy(idxs);
            dz      = dz(idxs);
            phi     = phi(idxs);
            theta   = theta(idxs);
            psi     = psi(idxs);
            p       = p(idxs);
            q       = q(idxs);
            r       = r(idxs);
            m       = m(idxs);
      


            % Calculate Costs
            L_F = u1.^2 + u2.^2 + u3.^2 + u4.^2;
            J_F = trapz(tc,L_F);


            J_path = J_F;
            J_term = 0;
            J_total = J_path + J_term;

        %     disp(['Force min cost is ',num2str(J_F)])
            disp(['Path Cost is  ', num2str(J_path)])
            disp(['Term cost is ', num2str(J_term)])
            disp(['Total cost is ', num2str(J_total)])

            % Save off outputs
            Jout(i,:) = [J_path,J_term,J_total];
            stateOut(:,:,i) = [tc',x',y',z',dx',dy',dz',phi',theta',psi',p',q',r',m'];
            ctrlOut(:,:,i) = [u1',u2',u3',u4'];
            runTimeOut(i) = timeToRun;
            stateFinal(i,:) = [xa(end),ya(end),za(end),dxa(end),dya(end),dza(end),phia(end),thetaa(end),psia(end),pa(end),qa(end),ra(end),ma(end)];
            
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
        %         plot(objective(1),objective(2),'c+','MarkerSize',10)
        %         plot(gridPoints(:,1),gridPoints(:,2),'.')
        %         plot(linspace(-100,100),surfFunction(linspace(-100,100)))
                xlabel('x[m]');ylabel('y[m]');zlabel('z[m]')
                legend('Starting Point','Trajectory','Ending Point','Objective','Surface','location','best')


                % Plot thrust profiles
                figure(2);
                subplot(2,2,1)
                plot(tc,u1,'g')
                hold on
                title('Controls')
                ylabel('u1 [N]')
                subplot(2,2,2)
                plot(tc,u2,'b')
                hold on
                ylabel('u2 [rad/s/s]')
                subplot(2,2,3)
                plot(tc,u3,'b')
                hold on
                ylabel('u3 [rad/s/s]')
                subplot(2,2,4)
                plot(tc,u4,'b')
                hold on
                ylabel('u4 [rad/s/s]')
                
                figure(3);
                subplot(2,2,1)
                hold on
                plot(tc,x,'b')
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
                
                figure(4);
                subplot(2,2,1)
                hold on
                plot(tc,dx,'b')
                title('Velocity vs Time')
                ylabel('xdot [m/s]')
                subplot(2,2,2)
                plot(tc,dy,'b')
                hold on
                ylabel('ydot [m/s]')
                subplot(2,2,3)
                plot(tc,dz,'b')
                hold on
                ylabel('zdot [m/s]')
                
                figure(5);
                subplot(2,2,1)
                hold on
                plot(tc,rad2deg(phi),'g')
                title('Euler Angles vs Time')
                ylabel('phi [deg]')
                subplot(2,2,2)
                plot(tc,rad2deg(theta),'b')
                hold on
                ylabel('theta [deg]')
                subplot(2,2,3)
                plot(tc,rad2deg(psi),'b')
                hold on
                ylabel('psi [deg]')

                figure(6);
                subplot(2,2,1)
                hold on
                plot(tc,rad2deg(p),'g')
                title('Body Rates vs Time')
                ylabel('p [deg/s]')
                subplot(2,2,2)
                plot(tc,rad2deg(q),'b')
                hold on
                ylabel('q [deg/s]')
                subplot(2,2,3)
                plot(tc,rad2deg(r),'b')
                hold on
                ylabel('r [deg/s]')


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
save(saveout,'surfFunctionOut','objectiveOut','Jout','stateOut','ctrlOut','runTimeOut','stateFinal','conf');
fprintf("\nProgram Complete!\n")
disp(['at ',datestr(now,'yyyymmdd_HHoMMSS')])

%% Solver Functions
function landervarsfun(sh, c)


    % Define States
    sh.addState('x');
    sh.addState('y');
    sh.addState('z');
    sh.addState('dx');
    sh.addState('dy');
    sh.addState('dz');
    sh.addState('phi');
    sh.addState('theta');
    sh.addState('psi');
    sh.addState('p');
    sh.addState('q');
    sh.addState('r');
    sh.addState('m');


    Fmax = 20;
    % Define Controls
    sh.addControl('u1', 'lb',   0, 'ub', 20);  % Force [N]
    sh.addControl('u2', 'lb', -20, 'ub', 20);  % Force [N]
    sh.addControl('u3', 'lb', -20, 'ub', 20);  % Force [N]
    sh.addControl('u4', 'lb', -20, 'ub', 20);  % Force [N]


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
    sh.setODE( 'x'    , x.dx);
    sh.setODE( 'y'    , x.dy);
    sh.setODE( 'z'    , x.dz);
    sh.setODE( 'dx'   , u.u1*(sin(x.phi)*sin(x.psi)+cos(x.phi)*sin(x.theta)*cos(x.psi)));
    sh.setODE( 'dy'   , u.u1*(-sin(x.phi)*cos(x.psi)+cos(x.phi)*sin(x.theta)*sin(x.psi)));
    sh.setODE( 'dz'   , u.u1*(cos(x.phi)*cos(x.theta)) - c.g);
    sh.setODE( 'phi'  , x.p + x.q*sin(x.phi)*tan(x.theta) + x.r*cos(x.phi)*tan(x.theta));
    sh.setODE( 'theta',       x.q*cos(x.phi)              - x.r*sin(x.phi));
    sh.setODE( 'psi'  ,       x.q*sin(x.phi)/cos(x.theta) + x.r*cos(x.phi)/cos(x.theta));
   
    sh.setODE( 'p'    , u.u2);
    sh.setODE( 'q'    , u.u3);
    sh.setODE( 'r'    , u.u4);
    
    sh.setODE( 'm'    , -sqrt((u.u1)^2 + (u.u2)^2 +(u.u3)^2 + (u.u4)^2)/(c.Isp*c.g0));


end

function landerpathcosts(ch,x,u,~)
    
    % Cost Function (thrust magnitude)
    ch.add(u.u1^2);
    ch.add(u.u2^2);
    ch.add(u.u3^2);
    ch.add(u.u4^2);


end



