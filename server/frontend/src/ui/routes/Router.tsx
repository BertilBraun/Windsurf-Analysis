import React from 'react'
import { useAuth } from '../auth/AuthProvider'
import { LoginPage } from '../screens/LoginPage'
import { SignupPage } from '../screens/SignupPage'
import { MainPage } from '../screens/MainPage'

type Route = 'login' | 'signup' | 'main'

function getInitialRoute(isAuthenticated: boolean): Route {
    if (!isAuthenticated) return 'login'
    return 'main'
}

export const Router: React.FC = () => {
    const { isAuthenticated } = useAuth()
    const [route, setRoute] = React.useState<Route>(() => getInitialRoute(isAuthenticated))

    React.useEffect(() => {
        setRoute(getInitialRoute(isAuthenticated))
    }, [isAuthenticated])

    const navigate = (r: Route) => setRoute(r)

    if (route === 'signup') {
        return <SignupPage onBackToLogin={() => navigate('login')} />
    }

    if (!isAuthenticated) {
        return <LoginPage onSignup={() => navigate('signup')} onSuccess={() => navigate('main')} />
    }

    return <MainPage />
}
